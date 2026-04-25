"""Downloads Speech Commands + MUSAN, mixes noise, writes splits to data/processed/."""

import csv
import os
import random
from pathlib import Path

import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from datasets import load_dataset


SAMPLE_RATE = 16000
CLIP_LENGTH = SAMPLE_RATE  # 1 second
SNR_LEVELS = (20, 10, 0, -5)
SEED = 42

# 10 target words. Everything else lands in "unknown"; "silence" is its own class.
TARGET_LABELS = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"}
LABEL_VOCAB = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABEL_VOCAB)}

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./data"))
PROCESSED_ROOT = DATA_ROOT / "processed"


def relabel(label):
    return label if label in TARGET_LABELS else "unknown"


def fit_to_length(waveform, target_len=CLIP_LENGTH):
    cur_len = waveform.shape[1]
    if cur_len > target_len:
        start = random.randint(0, cur_len - target_len)
        return waveform[:, start:start + target_len]
    if cur_len < target_len:
        # Loop the clip rather than zero-pad, so silence doesn't bleed in
        repeats = (target_len // cur_len) + 1
        waveform = waveform.repeat(1, repeats)
    return waveform[:, :target_len]


def compute_rms(waveform, eps=1e-8):
    return torch.sqrt(torch.mean(waveform ** 2) + eps)


def mix_at_snr(speech, noise, snr_db):
    # Standard SNR-controlled additive mix. Clamp at the end since the sum can clip.
    speech_rms = compute_rms(speech)
    noise_rms = compute_rms(noise)
    desired_noise_rms = speech_rms / (10 ** (snr_db / 20))
    scale = desired_noise_rms / (noise_rms + 1e-8)
    return torch.clamp(speech + scale * noise, -1.0, 1.0)


def create_silence_clips(silence_dir):
    # The dataset ships ~6 long background recordings; chop them into 1 s slices.
    silence_clips = []
    for fname in sorted(os.listdir(silence_dir)):
        if not fname.endswith(".wav"):
            continue
        waveform, file_sr = torchaudio.load(os.path.join(silence_dir, fname))
        if file_sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(file_sr, SAMPLE_RATE)(waveform)

        file_name = fname.replace(".wav", "")
        num_clips = waveform.shape[1] // CLIP_LENGTH
        for i in range(num_clips):
            clip = waveform[:, i * CLIP_LENGTH:(i + 1) * CLIP_LENGTH]
            silence_clips.append((clip, file_name))

    print(f"Built {len(silence_clips)} silence clips")
    return silence_clips


def build_musan_cache(musan_ds):
    # Pull every MUSAN clip into memory once, resampled. Decoding per epoch is brutal.
    cache = []
    train = musan_ds["train"]
    for i in range(len(train)):
        item = train[i]
        waveform = torch.tensor(item["audio"]["array"]).float().unsqueeze(0)
        sr = item["audio"]["sampling_rate"]
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        cache.append({
            "waveform": waveform,
            "category": train.features["label"].names[item["label"]],
        })
    print(f"Cached {len(cache)} MUSAN samples")
    return cache


def inject_noise(speech, musan_waveform, snr_db):
    noise = fit_to_length(musan_waveform)
    speech = fit_to_length(speech)
    return mix_at_snr(speech, noise, snr_db)


def export_split(
    speech_dataset,
    split_name,
    musan_cache,
    silence_clips,
    inject=False,
    fixed_snr=None,
    seed=SEED,
):
    rng = random.Random(seed)
    out_dir = PROCESSED_ROOT / split_name
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for i in range(len(speech_dataset)):
        waveform, _, raw_label, speaker_id, utterance_number = speech_dataset[i]
        label = relabel(raw_label)

        snr_used = None
        musan_category = None
        musan_idx = None

        # Don't mix noise into silence — the whole point is that the model sees clean silence
        if inject and label != "silence":
            musan_idx = rng.randint(0, len(musan_cache) - 1)
            musan_item = musan_cache[musan_idx]
            musan_category = musan_item["category"]
            snr_used = fixed_snr if fixed_snr is not None else rng.choice(list(SNR_LEVELS))
            waveform = inject_noise(waveform, musan_item["waveform"], snr_used)

        waveform = fit_to_length(waveform)
        filepath = audio_dir / f"{split_name}_{i:06d}.wav"
        torchaudio.save(str(filepath), waveform, SAMPLE_RATE)

        rows.append({
            "file_path": str(filepath),
            "split": split_name,
            "label": label,
            "label_idx": LABEL_TO_IDX[label],
            "raw_label": raw_label,
            "speaker_id": speaker_id,
            "utterance_number": utterance_number,
            "is_silence": 0,
            "musan_category": musan_category,
            "musan_idx": musan_idx,
            "snr_db": snr_used,
        })

    for j, (clip, file_name) in enumerate(silence_clips):
        clip = fit_to_length(clip)
        filepath = audio_dir / f"{split_name}_silence_{j:06d}.wav"
        torchaudio.save(str(filepath), clip, SAMPLE_RATE)
        rows.append({
            "file_path": str(filepath),
            "split": split_name,
            "label": "silence",
            "label_idx": LABEL_TO_IDX["silence"],
            "raw_label": "silence",
            "speaker_id": file_name,
            "utterance_number": j,
            "is_silence": 1,
            "musan_category": None,
            "musan_idx": None,
            "snr_db": None,
        })

    manifest_path = out_dir / "metadata.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {split_name}: {len(rows)} examples → {manifest_path}")


def snr_folder(prefix, snr):
    # "-5db" breaks shell globbing in some places; m5db is safer
    return f"{prefix}_{snr}db" if snr >= 0 else f"{prefix}_m5db"


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    print("Downloading Speech Commands v0.02...")
    train_set = SPEECHCOMMANDS(str(DATA_ROOT), download=True, subset="training")
    val_set = SPEECHCOMMANDS(str(DATA_ROOT), download=True, subset="validation")
    test_set = SPEECHCOMMANDS(str(DATA_ROOT), download=True, subset="testing")
    print(f"  train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    silence_dir = DATA_ROOT / "SpeechCommands" / "speech_commands_v0.02" / "_background_noise_"
    silence_clips = create_silence_clips(silence_dir)

    print("Loading MUSAN from HuggingFace...")
    musan_ds = load_dataset("Aynursusuz/musan-audio-dataset")
    musan_cache = build_musan_cache(musan_ds)

    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    # Train: one clean, one with random SNR per clip, plus one per fixed SNR
    print("\nExporting train splits...")
    export_split(train_set, "train_clean", musan_cache, silence_clips, inject=False)
    export_split(train_set, "train_noisy", musan_cache, silence_clips, inject=True)
    for snr in SNR_LEVELS:
        export_split(train_set, snr_folder("train", snr), musan_cache, silence_clips, inject=True, fixed_snr=snr)

    # Val/test: clean baseline + one frozen split per SNR for consistent evaluation
    print("\nExporting val splits...")
    export_split(val_set, "val_clean", musan_cache, silence_clips, inject=False)
    for snr in SNR_LEVELS:
        export_split(val_set, snr_folder("val", snr), musan_cache, silence_clips, inject=True, fixed_snr=snr)

    print("\nExporting test splits...")
    export_split(test_set, "test_clean", musan_cache, silence_clips, inject=False)
    for snr in SNR_LEVELS:
        export_split(test_set, snr_folder("test", snr), musan_cache, silence_clips, inject=True, fixed_snr=snr)

    print(f"\nDone. All splits written to {PROCESSED_ROOT}")


if __name__ == "__main__":
    main()