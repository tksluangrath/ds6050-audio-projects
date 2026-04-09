import argparse
import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

from audio_preprocessing import get_log_mel_spectrogram

SAMPLE_RATE = 16000

MODEL_DEFAULTS = {
    "cnn": {"lr": 1e-3, "n_mels": 40,  "n_fft": 400},
    "ast": {"lr": 1e-5, "n_mels": 128, "n_fft": 1024},
}
AST_WARMUP_EPOCHS = 5


# Tee: mirror all print() output to a log file

class Tee:
    def __init__(self, filepath):
        self.file = open(filepath, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout


# Dataset

class SpeechCommandsDataset(Dataset):
    def __init__(self, split: str, data_root: str, filter_method: str, n_mels: int = 40, n_fft: int = 400):
        metadata_csv = os.path.join(data_root, "data", "processed", split, "metadata.csv")
        self.df = pd.read_csv(metadata_csv)
        self.data_root = data_root
        self.filter_method = filter_method
        self.n_mels = n_mels
        self.n_fft = n_fft

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.data_root, row["file_path"])

        data, sr = sf.read(audio_path, dtype="float32")
        waveform = torch.tensor(data, dtype=torch.float32)

        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)

        if waveform.shape[0] < SAMPLE_RATE:
            waveform = torch.nn.functional.pad(waveform, (0, SAMPLE_RATE - waveform.shape[0]))
        else:
            waveform = waveform[:SAMPLE_RATE]

        spec = get_log_mel_spectrogram(waveform, filter_method=self.filter_method, n_mels=self.n_mels, n_fft=self.n_fft)
        label = torch.tensor(int(row["label_idx"]), dtype=torch.long)
        return spec, label


def make_loader(split, data_root, filter_method, batch_size, shuffle, device, n_mels=40, n_fft=400):
    dataset = SpeechCommandsDataset(split, data_root, filter_method, n_mels, n_fft)

    def collate_fn(batch):
        specs, labels = zip(*batch)
        return torch.stack(specs).to(device), torch.stack(labels).to(device)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


# Scheduler

def build_scheduler(optimizer, model_type, epochs):
    if model_type != "ast":
        return None
    warmup_epochs = min(AST_WARMUP_EPOCHS, epochs // 3)
    cosine_epochs = max(epochs - warmup_epochs, 1)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-7)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


# Main runner

def run(model_cls, train_fn, evaluate_fn, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve model-specific defaults
    defaults = MODEL_DEFAULTS[args.model]
    n_mels = defaults["n_mels"]
    n_fft = defaults["n_fft"]
    lr = args.lr if args.lr is not None else defaults["lr"]

    # Set up output directory (consistent name regardless of eval_only)
    run_name = f"{args.model}_{args.train_split}_{args.val_split}_{args.filter_method}"
    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = os.path.join(out_dir, "best_model.pt")
    if args.eval_only:
        log_name = f"eval_{args.test_split}.txt" if args.test_split else "eval_log.txt"
    else:
        log_name = "log.txt"
    log_path = os.path.join(out_dir, log_name)

    tee = Tee(log_path)
    sys.stdout = tee

    print(f"Using device: {device}")
    print(f"Run name:      {run_name}")
    print(f"Output dir:    {out_dir}")
    print(f"Model:         {args.model}")
    print(f"Train split:   {args.train_split}")
    print(f"Val split:     {args.val_split}")
    print(f"Test split:    {args.test_split or 'none'}")
    print(f"Filter method: {args.filter_method}")
    print(f"Mel bins:      {n_mels}  |  N_FFT: {n_fft}")
    print(f"Mode:          {'eval only' if args.eval_only else 'train + eval'}")
    if not args.eval_only:
        scheduler_type = f"warmup({min(AST_WARMUP_EPOCHS, args.epochs // 3)}) + cosine" if args.model == "ast" else "none"
        print(f"Epochs:        {args.epochs}  |  Batch size: {args.batch_size}  |  LR: {lr}  |  Clip: {args.clip}")
        print(f"Scheduler:     {scheduler_type}")
    print()

    criterion = nn.CrossEntropyLoss()
    model = model_cls().to(device)

    # ── Eval-only mode ────────────────────────────────────────────────────────
    if args.eval_only:
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: No checkpoint found at {checkpoint_path}")
            print("Train the model first without --eval_only.")
            tee.close()
            return

        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        if args.test_split:
            print(f"Loading {args.test_split} data...")
            test_loader = make_loader(
                args.test_split, args.data_root, args.filter_method,
                args.batch_size, shuffle=False, device=device, n_mels=n_mels, n_fft=n_fft,
            )
            test_loss, test_acc = evaluate_fn(model, test_loader, criterion)
            print(f"Test split: {args.test_split} | Loss: {test_loss:.3f} | Acc: {test_acc*100:.2f}%")
        else:
            print("Loading validation data...")
            val_loader = make_loader(
                args.val_split, args.data_root, args.filter_method,
                args.batch_size, shuffle=False, device=device, n_mels=n_mels, n_fft=n_fft,
            )
            val_loss, val_acc = evaluate_fn(model, val_loader, criterion)
            print(f"Val split:  {args.val_split} | Loss: {val_loss:.3f} | Acc: {val_acc*100:.2f}%")

        print("\nDone.")
        tee.close()
        return

    # ── Training mode ─────────────────────────────────────────────────────────
    print("Loading training data...")
    train_loader = make_loader(
        args.train_split, args.data_root, args.filter_method,
        args.batch_size, shuffle=True, device=device, n_mels=n_mels, n_fft=n_fft,
    )

    print("Loading validation data...")
    val_loader = make_loader(
        args.val_split, args.data_root, args.filter_method,
        args.batch_size, shuffle=False, device=device, n_mels=n_mels, n_fft=n_fft,
    )

    test_loader = None
    if args.test_split:
        print("Loading test data...")
        test_loader = make_loader(
            args.test_split, args.data_root, args.filter_method,
            args.batch_size, shuffle=False, device=device, n_mels=n_mels, n_fft=n_fft,
        )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = build_scheduler(optimizer, args.model, args.epochs)
    best_val_loss = float("inf")

    with open(os.path.join(out_dir, "config.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write(f"lr_resolved: {lr}\n")
        f.write(f"n_mels: {n_mels}\n")

    for epoch in range(args.epochs):
        start = time.time()
        train_loss = train_fn(model, train_loader, optimizer, criterion, args.clip, scheduler)
        val_loss, val_acc = evaluate_fn(model, val_loader, criterion)
        elapsed = time.time() - start

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            marker = " *"
        else:
            marker = ""

        lr_str = f" | LR: {optimizer.param_groups[0]['lr']:.2e}" if scheduler else ""
        print(f"Epoch {epoch+1:02} | {elapsed:.1f}s | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc*100:.2f}%{lr_str}{marker}")

    if test_loader is not None:
        print(f"\nLoading best checkpoint for test evaluation...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        test_loss, test_acc = evaluate_fn(model, test_loader, criterion)
        print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")

    print("\nDone.")
    tee.close()


# CLI

def parse_args():
    parser = argparse.ArgumentParser(description="Train a keyword spotting model.")
    parser.add_argument("--model", choices=["cnn", "ast"], default="cnn")
    parser.add_argument("--train_split", default="train_noisy")
    parser.add_argument("--val_split", default="val_clean")
    parser.add_argument("--test_split", default=None)
    parser.add_argument("--filter_method", choices=["none", "bandpass", "spectral_gating"], default="none")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--data_root", default=".")
    parser.add_argument("--eval_only", action="store_true")
    return parser.parse_args()


def main():
    from train import train
    from evaluate import evaluate
    from CNN_Model import KeywordSpottingCNN
    from AST import AST

    args = parse_args()
    model_cls = KeywordSpottingCNN if args.model == "cnn" else AST
    run(model_cls, train, evaluate, args)


if __name__ == "__main__":
    main()
