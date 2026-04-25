# Frequency-Based Speech Isolation for Keyword Spotting

> **Does frequency-domain noise suppression improve keyword spotting robustness under real-world noise conditions?**


This repo holds the code, experiments, and paper for Group 09's final project in DS 6050 (Deep Learning) at the University of Virginia. We test whether spectral preprocessing (spectral gating and a fixed bandpass filter) makes compact keyword spotting (KWS) models more robust at different SNRs. 


**Authors:** Terrance Luangrath, Samantha Asefi, Lucas Anderson, Tomas Tsega

---

## Project Overview

Voice assistants struggle in noisy rooms. HVAC hum, crowd babble, music in the background. We wanted to know whether suppressing noise in the frequency domain (before the Mel spectrogram step) actually helps, and whether a compact CNN and an Audio Spectrogram Transformer (AST) react the same way to it. They don't.

**What we found:**

**Research question:** Can explicit frequency-domain noise suppression, applied as a deterministic preprocessing step, improve keyword spotting accuracy across varying noise types and signal-to-noise ratios?

---

## Dataset

**Google Speech Commands v0.02** — 105,829 one-second clips at 16 kHz. We use the 10 standard commands (`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`) plus `unknown` (the other 25 words collapsed) and `silence` for 12 classes total, loaded via HuggingFace `datasets` (pinned `< 3.0`). The official `validation_list.txt` and `testing_list.txt` splits keep our numbers comparable to published results. The `silence` class is built from background-noise clips chunked into overlapping 1 s windows with a 0.5 s hop.

**MUSAN** — used for noise injection in training and evaluation. Download from [hugging face](https://huggingface.co/datasets/Aynursusuz/musan-audio-dataset) and place under `data/musan/`. We use three MUSAN categories (babble from the speech subset, music, ambient noise) plus synthetic Gaussian white noise, mixed at SNRs of {20, 10, 0, −5} dB.

---

## Preprocessing Pipelines

Both models consume log-Mel spectrograms, but with different specs (the AST needs higher resolution to populate its 128-bin filterbank and match its pretrained patch grid):


| Model | Window | FFT | Mel bins | Hop |
|-------|--------|-----|----------|-----|
| CNN | 25 ms | 400 | 40 | 10 ms |
| AST | 64 ms | 1024 | 128 | 10 ms |


**A - Bandpass only (baseline).** Cascaded biquad IIR with a 300 Hz high-pass and 3400 Hz low-pass on the raw waveform. No tunable parameters. This is what we compare spectral gating against.

**B - Spectral gating.** Estimate the noise power spectrum from the first five STFT frames (where energy is minimal) and over-subtract:

$$|\hat{S}(f, \tau)|^2 = \max\left(|X(f, \tau)|^2 - \omega|\hat{N}(f)|^2,\ \delta \cdot |X(f, \tau)|^2\right)$$

with $\omega = 1.5$ (over-subtraction strength) and $\delta = 0.02$ (spectral floor, prevents musical noise). Both selected by sweeping $\omega\in\{1.0, 1.5, 2.0\}$ and $\delta\in\{0.01, 0.02, 0.05\}$ on the validation set.

---

## Models

**Compact Convolutional Neural Networks (CNN).** Three Conv blocks (16, 32, 64 filters, 3x3, ReLU, same padding) each followed by $2\times2$ max-pool $\rightarrow$ flatten $\rightarrow$ FC (128) with dropout ($p=0.5$) $\rightarrow$ 12-class output. Trained from scratch.

**Audio Spectrogram Transformer (AST).** DeiT-Base (distilled) backbone, pretrained on ImageNet-21k. Patch size 16, stride (10, 10). The 3-channel patch projection is averaged across channels for single-channel audio input; positional embeddings are bilinearly interpolated from DeiT's $14\times14$ grid to the $12\times9$ patch grid produced by 128 mel bins $\times$ 101 time frames. The classification head is a fresh `Linear(768, 12)` on the `[CLS]` token. We omit AudioSet pretraining, so all features come from natural images rather than audio.

--

## Noise Augmentation

Noise is mixed in on-the-fly per epoch (not pre-computed), so the model sees a different sample every pass, and we don't have the store augmented copies. Five test splits are held out separately: one clean, one per SNR level.

| Condition | SNR |
|-----------|-----|
| Clean | — |
| Mild | 20 dB |
| Moderate | 10 dB |
| Severe | 0 dB |
| Extreme | −5 dB |

```python
alpha = sqrt(P_speech / (P_noise * 10 ** (snr_db / 10)))
noisy = speech + alpha * noise
```

---

## Experiments & Results

We trained eight configurations: two architectures (CNN, AST) $\times$ two training regimes (clean, noisy) $\times$ three preprocessing filters (none, bandpass, spectral gating). Evaluation runs through `model/evaluate.py`; per-class F1 and the -5 dB confusion matrices come from `model/analysis.py`. 

**Test accuracy (%) on noisy-trained models — best in bold:**


--- 

## Setup & Reproducing Results

Python 3.10+, two NVIDIA GPUs recommended (we used 2x RTX 3090 on UVA Rivanna).

```bash
git clone https://github.com/tksluangrath/noise-robust-keyword-spotting/blob/main/README.md
cd https://github.com/tksluangrath/noise-robust-keyword-spotting/blob/main/README.md
pip install -r requirements.txt

# Download dataset
python data/download.py

# Tune spectral-gating params (required before runner.py)
python model/sweep.py --model cnn
python model/sweep.py --model ast

# Train (one GPU per model)
CUDA_VISIBLE_DEVICES=0 python model/runner.py --model cnn --seed 6050
CUDA_VISIBLE_DEVICES=1 python model/runner.py --model ast --seed 6050
```

Each configuration writes to `runs/{model}_{splits}_{filter}_seed{N}/` - checkpoint, log, config, and the per-splt analysis from `get_analysis`.

Results land in `results/` as CSSV and PNG.

The full paper is at [`paper/G09_paper.pdf`](paper/G09_paper.pdf) (built on Overleaf).

---