# Sequence Layers

Note: This is not an officially supported Google product

## Overview

A library for sequence modeling in Jax and TensorFlow 2, enabling easy creation
of sequence models that can be executed both layer-by-layer (e.g. teacher forced
training) and step-by-step (e.g. autoregressive sampling).

A key feature of the library is that layers support streaming (step-by-step)
operation. To achieve this, every layer has a notion of state when and a `step`
function in addition to the typical layer-wise processing feature found in other
libraries like Keras. When layers support a `step` method, their `layer` method
produces identical results for the same sequence of input blocks enabling easy
switching between step-wise and layer-wise processing depending on the use case.

**Note:** Only Jax support is installed by default. Use
`pip install sequence_layers[tensorflow]` for TensorFlow.

## Goals

Increased development velocity for both research and production applications of
sequence modeling.

*   Support for layer-by-layer and step-by-step processing in a single
    implementation.
*   Declarative API.
*   Composable, thin abstractions.
*   Easy mix-and-match of popular sequence modeling paradigms (convolutional,
    recurrent, attention architectures).
*   A quick path to deployment with tf.lite support for every layer.
*   Tracking of invalid timesteps (those computed from padding).

## PyTorch BEST-RQ Example

A simple example to pretrain the [BEST-RQ] model on the Librispeech dataset is
provided in `pytorch_examples/best_rq_librispeech.py`. This implementation
mirrors the Conformer encoder from `sequence_layers/tensorflow/examples`
and does not rely on external BEST‑RQ packages. Only `torchaudio` is required
for dataset loading. Default hyper‑parameters roughly follow the paper, using an
8192 entry codebook and 16‑d projections. Before training, run
`python pytorch_examples/compute_cmvn.py` to compute global CMVN statistics.
Inputs are normalized using these statistics and masked waveform segments are
replaced with Gaussian noise. The dataloader returns each batch alongside the
original lengths, which must be passed to the training wrapper. The default
batch size is 16. Run the example with:

```bash
python pytorch_examples/best_rq_librispeech.py --download
```
