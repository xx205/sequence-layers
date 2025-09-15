import argparse
import json
import math

import torch
import torchaudio


def compute_cmvn(dataset, target_sr: int):
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr, n_fft=400, hop_length=160, n_mels=80
    )
    db = torchaudio.transforms.AmplitudeToDB()

    total_sum = 0.0
    total_sq_sum = 0.0
    total_frames = 0

    for waveform, orig_sr, *_ in dataset:
        waveform = waveform.squeeze(0)
        if orig_sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, target_sr)
        feats = db(melspec(waveform))
        total_sum += feats.sum().item()
        total_sq_sum += feats.pow(2).sum().item()
        total_frames += feats.numel()

    mean = total_sum / total_frames
    std = math.sqrt(total_sq_sum / total_frames - mean ** 2)
    return mean, std


def main(args: argparse.Namespace):
    dataset = torchaudio.datasets.LIBRISPEECH(
        args.root, url=args.subset, download=args.download
    )
    mean, std = compute_cmvn(dataset, args.sample_rate)
    with open(args.output, "w") as f:
        json.dump({"mean": mean, "std": std}, f)
    print(f"Saved CMVN statistics to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--subset", type=str, default="train-clean-100")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--output", type=str, default="cmvn_stats.json")
    args = parser.parse_args()
    main(args)
