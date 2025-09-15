import argparse
import argparse
import os
import math
import json
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader


class AddTimingSignal(nn.Module):
    """Adds a sinusoidal timing signal to the input."""

    def __init__(self, dim: int, timescale: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.timescale = timescale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.arange(x.size(1), device=x.device, dtype=x.dtype)
        freqs = torch.exp(
            -math.log(self.timescale)
            * torch.arange(0, self.dim, 2, device=x.device, dtype=x.dtype)
            / self.dim
        )
        angles = torch.einsum("n,d->nd", t, freqs)
        signal = torch.zeros((x.size(1), self.dim), device=x.device, dtype=x.dtype)
        signal[:, 0::2] = torch.sin(angles)
        signal[:, 1::2] = torch.cos(angles)
        return x + signal.unsqueeze(0)


class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + 0.5 * self.ffn(self.norm(x))
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        return x


class MultiHeadedSelfAttention(nn.Module):
    """Self-attention with relative positional bias."""

    def __init__(self, dim: int, num_heads: int, max_horizon: int, dropout: float):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_horizon = max_horizon

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        if max_horizon >= 0:
            # Relative positional embeddings for 0..max_horizon timesteps.
            self.rel_pos = nn.Parameter(
                torch.randn(max_horizon + 1, num_heads, self.head_dim)
            )
        else:
            self.rel_pos = None

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        res = x
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        b, t, _ = q.size()

        def reshape(tensor):
            return tensor.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.rel_pos is not None:
            # Relative position bias for past timesteps.
            pos = torch.arange(t, device=x.device)
            rel = pos.view(1, -1) - pos.view(-1, 1)
            rel = (-rel).clamp_(0, self.max_horizon)
            rel_bias = self.rel_pos[rel]
            rel_scores = torch.einsum("bhqd,tthd->bhqt", q, rel_bias)
            rel_scores = rel_scores / math.sqrt(self.head_dim)
            scores = scores + rel_scores

        if self.max_horizon >= 0:
            pos = torch.arange(t, device=x.device)
            rel = pos.unsqueeze(1) - pos.unsqueeze(0)
            causal_mask = (rel < 0) | (rel > self.max_horizon)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if key_padding_mask is not None:
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_mask, float("-inf"))
            v = v.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(-1), 0.0)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        out = self.out_proj(out)
        out = res + self.dropout(out)
        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return out


class ConvolutionModule(nn.Module):
    def __init__(self, dim: int, dropout: float, kernel_size: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pw_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.depthwise = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=0, groups=dim
        )
        self.bn = nn.BatchNorm1d(dim)
        self.pw_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        res = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.pw_conv1(x)
        x = F.glu(x, dim=1)
        padding = self.depthwise.kernel_size[0] - 1
        x = F.pad(x, (padding, 0))
        x = self.depthwise(x)
        x = self.bn(x)
        x = F.silu(x)
        x = self.pw_conv2(x)
        x = self.dropout(x.transpose(1, 2))
        x = res + x
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        return x


class ConformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_horizon: int, dropout: float):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, dropout)
        self.mha = MultiHeadedSelfAttention(dim, num_heads, max_horizon, dropout)
        self.conv = ConvolutionModule(dim, dropout)
        self.ff2 = FeedForwardModule(dim, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.ff1(x, mask)
        x = self.mha(x, key_padding_mask=mask)
        x = self.conv(x, mask)
        x = self.ff2(x, mask)
        x = self.norm(x)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        return x


class ConvolutionSubsampling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, T/4, C)
        x = x.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2)


class ConformerEncoder(nn.Module):
    def __init__(self, dim: int, depth: int, max_horizon: int, dropout: float = 0.1):
        super().__init__()
        self.subsample = ConvolutionSubsampling(dim)
        self.input_linear = nn.Linear(dim, dim)
        self.add_timing = AddTimingSignal(dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [ConformerBlock(dim, 8, max_horizon, dropout) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.subsample(x)
        x = self.input_linear(x)
        x = self.add_timing(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, mask)
        return x


class RandomProjectionQuantizer(nn.Module):
    """Random-projection quantizer used by BEST-RQ."""

    def __init__(self, num_codes: int, proj_dim: int, input_dim: int, stack: int = 4):
        super().__init__()
        self.stack = stack
        # Random projection matrix and codebook are fixed (no gradients).
        self.register_buffer("proj", torch.empty(input_dim * stack, proj_dim))
        self.register_buffer("codebook", torch.randn(num_codes, proj_dim))
        with torch.no_grad():
            nn.init.xavier_uniform_(self.proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        b, t, c = x.shape
        if self.stack > 1:
            pad = (self.stack - t % self.stack) % self.stack
            if pad:
                x = F.pad(x, (0, 0, 0, pad))
                t = x.size(1)
            x = x.view(b, t // self.stack, c * self.stack)
        x_proj = F.normalize(torch.matmul(x, self.proj), dim=-1)
        codebook = F.normalize(self.codebook, dim=-1)
        diff = x_proj.unsqueeze(-2) - codebook.unsqueeze(0)
        dist = (diff ** 2).sum(-1)
        return dist.argmin(-1)


class BestRQ(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        sample_rate: int,
        model_dim: int,
        depth: int,
    ):
        super().__init__()

        self.pad_id = 0
        self.eos_id = codebook_size + 1

        self.sample_rate = sample_rate
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=80
        )
        self.db = torchaudio.transforms.AmplitudeToDB()
        self.quantizer = RandomProjectionQuantizer(
            codebook_size, codebook_dim, 80, stack=4
        )

        self.register_buffer("global_mean", torch.tensor(0.0))
        self.register_buffer("global_std", torch.tensor(1.0))

        self.feat_proj = nn.Linear(80, model_dim)

        # "max_horizon" bounds the accessible past context similar to the TF
        # version. -1 means no limit.
        self.encoder = ConformerEncoder(model_dim, depth, max_horizon=-1)
        self.to_logits = nn.Linear(model_dim, codebook_size + 2)

    def encode_features(self, wav: torch.Tensor) -> torch.Tensor:
        self.melspec = self.melspec.to(wav.device)
        self.db = self.db.to(wav.device)
        feats = self.melspec(wav)
        feats = self.db(feats)
        feats = (feats - self.global_mean) / self.global_std
        return feats.transpose(1, 2)

    def lengths_to_encoder(self, lengths: torch.Tensor) -> torch.Tensor:
        frame_lens = (lengths - self.melspec.n_fft) // self.melspec.hop_length + 1
        frame_lens = (frame_lens - 1) // 2
        frame_lens = (frame_lens - 1) // 2
        return frame_lens

    def quantize(self, feats: torch.Tensor) -> torch.Tensor:
        labels = self.quantizer(feats)
        return labels + 1

    def forward(self, feats: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.feat_proj(feats)
        x = self.encoder(x, mask)
        return self.to_logits(x)


class BestRQPretrainWrapper(nn.Module):
    def __init__(self, model: BestRQ, mask_prob: float = 0.01, mask_span_ms: int = 400):
        super().__init__()
        self.model = model
        self.mask_prob = mask_prob
        self.mask_span_ms = mask_span_ms

    def forward(self, wav: torch.Tensor, lengths: torch.Tensor):
        with torch.no_grad():
            feats = self.model.encode_features(wav)
            labels = self.model.quantize(feats)

        b, t, _ = feats.shape
        feat_lengths = (lengths - self.model.melspec.n_fft) // self.model.melspec.hop_length + 1
        padding_mask = lengths_to_mask(feat_lengths, t)

        mask = (torch.rand(b, t, device=feats.device) < self.mask_prob) & (~padding_mask)
        span = int(self.mask_span_ms / 1000 * self.model.sample_rate / self.model.melspec.hop_length)
        if span > 1:
            padding = span - 1
            pad_left = (span - 1) // 2
            pad_right = padding - pad_left
            mask = F.pad(mask.unsqueeze(1).float(), (pad_left, pad_right), value=0)
            mask = F.max_pool1d(
                mask,
                kernel_size=span,
                stride=1,
                padding=0,
            ).squeeze(1).bool()
        noise = torch.randn_like(feats) * 0.1
        feats = torch.where(mask.unsqueeze(-1), noise, feats)

        enc_lengths = self.model.lengths_to_encoder(lengths)
        label_mask = lengths_to_mask(enc_lengths, labels.size(1))
        labels = labels.masked_fill(label_mask, self.model.pad_id)
        enc_t = (t - 1) // 2
        enc_t = (enc_t - 1) // 2
        enc_mask = lengths_to_mask(enc_lengths, enc_t)

        logits = self.model(feats, enc_mask)
        n_logits = logits.size(1)
        labels = labels[:, :n_logits]
        loss = F.cross_entropy(
            logits.transpose(1, 2), labels, ignore_index=self.model.pad_id
        )
        return loss, logits


def collate(batch, target_sr: int):
    """Collate a batch of variable length waveforms.

    Each item from ``LIBRISPEECH`` provides its own sample rate, so we resample
    individually when necessary.
    """
    waveforms = []
    lengths = []
    for waveform, orig_sr, *_ in batch:
        waveform = waveform.squeeze(0)
        if orig_sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, target_sr)
        lengths.append(waveform.size(0))
        waveforms.append(waveform)

    waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    lengths = torch.tensor(lengths)
    return waveforms, lengths


def lengths_to_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    if max_len is None:
        max_len = int(lengths.max())
    seq = torch.arange(max_len, device=lengths.device)
    return seq.unsqueeze(0) >= lengths.unsqueeze(1)


def main(args):
    if not os.path.exists(args.cmvn_path):
        raise FileNotFoundError(
            f"CMVN statistics file not found at {args.cmvn_path}. "
            f"Please run `compute_cmvn.py` first."
        )
    with open(args.cmvn_path, "r") as f:
        stats = json.load(f)
    mean = stats["mean"]
    std = stats["std"]

    dataset = torchaudio.datasets.LIBRISPEECH(
        args.root, url=args.subset, download=args.download
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, args.sample_rate),
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    model = BestRQ(
        codebook_size=args.codebook_size,
        codebook_dim=args.codebook_dim,
        sample_rate=args.sample_rate,
        model_dim=args.model_dim,
        depth=args.depth,
    )
    model.global_mean.fill_(mean)
    model.global_std.fill_(std)
    wrapper = BestRQPretrainWrapper(
        model, mask_prob=args.mask_prob, mask_span_ms=args.mask_span_ms
    )
    wrapper.to(args.device)
    optimizer = torch.optim.Adam(wrapper.parameters(), lr=args.lr)

    model.train()
    wrapper.train()
    for step, (batch, lengths) in enumerate(loader, start=1):
        batch = batch.to(args.device)
        lengths = lengths.to(args.device)
        loss, _ = wrapper(batch, lengths)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % args.log_interval == 0:
            print(f"step {step}: loss {loss.item():.4f}")
        if step == args.num_steps:
            break
    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save({"model": model.state_dict()}, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--subset", type=str, default="train-clean-100")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument("--codebook-dim", type=int, default=16)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--mask-prob", type=float, default=0.01)
    parser.add_argument("--mask-span-ms", type=int, default=400)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-path", type=str, default="best_rq.ckpt")
    parser.add_argument(
        "--cmvn-path",
        type=str,
        default="cmvn_stats.json",
        help="Path to the CMVN statistics file.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
