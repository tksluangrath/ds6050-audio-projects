"""
Sweep spectral gating hyperparameters (omega, delta) for the AST model.
GOAL: Evaluate a noisy-trained AST checkpoints on the clean val split.
"""

import sys, os, csv
sys.path.insert(0, os.path.dirname(__file__))

import itertools, torch, torch.nn as nn

import torch.utils.data as _data


# Force num_worker=0 everywhere. This prevents the cluster from hitting fork() memory limits
class _SingleProcDataLoader(_data.DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        super().__init__(*args, **kwargs)
_data.DataLoader = _SingleProcDataLoader

from runner import make_loader, MODEL_DEFAULTS, Tee
from evaluate import evaluate
from AST import AST
from CNN import KeywordSpottingCNN as CNN

# Paths and sweep config
ROOT = os.path.join(os.path.dirname(__file__), "..")
RUNS_DIR = os.path.join(ROOT, "runs")
omegas = [1.0, 1.5, 2.0]
deltas = [0.01, 0.02, 0.05]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

def find_ckpt(prefix):
    """Find the best_model.pt under runs/."""
    for entry in os.listdir(RUNS_DIR):
        if entry.startswith(prefix):
            candidate = os.path.join(RUNS_DIR, entry, "best_model.pt")
            if os.path.exists(candidate):
                return candidate
    raise FileNotFoundError(f"No checkpoint found with prefix {prefix}")


def sweep(name, model_cls, ckpt_prefix):
    """Run the omega/delta grid for one model and print results."""
    ckpt = find_ckpt(ckpt_prefix)
    n_mels = MODEL_DEFAULTS[name]["n_mels"]
    n_fft = MODEL_DEFAULTS[name]["n_fft"]

    # Create output dir and tee stdout to log.txt
    run_name = f"sweep_spectral_gating_{name}"
    out_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(out_dir, exist_ok = True)
    tee = Tee(os.path.join(out_dir, "log.txt"))

    # Initiate model and load weights
    model = model_cls().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    print(f"\n=== {name.upper()} ===")

    # Evaluate every (omega, delta) combination
    results = []
    for omega, delta in itertools.product(omegas, deltas):
        loader = make_loader(
            "val_clean",
            ROOT,
            "spectral_gating",
            64,
            False,
            n_mels = n_mels,
            n_fft = n_fft,
            omega = omega,
            delta = delta,
            num_workers = 0
        )
        _, acc, _, _ = evaluate(model, loader, criterion)
        results.append((omega, delta, acc))
        print(f"omega={omega:.2f} | delta={delta:.2f} | val acc: {acc*100:.2f}%")

    # Report best config
    best = max(results, key = lambda x: x[2])
    print(f"Best: omega={best[0]:.2f} | delta={best[1]:.2f} | val acc: {best[2]*100:.2f}%")
    print("Done.")

    # Save grid as CSV
    csv_path = os.path.join(out_dir, f"{name}_results.csv")
    with open(csv_path, "w", newline ="") as f:
        writer = csv.writer(f)
        writer.writerow(["omega", "delta", "val_acc"])
        for omega, delta, acc in results:
            writer.writerow([omega, delta, f"{acc:.4f}"])
                
    tee.close()
    # Free GPU memory before the next model loads
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    sweep("cnn", CNN, "cnn_train_noisy_val_clean_none")
    sweep("ast", AST, "ast_train_noisy_val_clean_none")


    

