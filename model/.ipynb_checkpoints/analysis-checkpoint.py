import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

LABEL_VOCAB = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]

def get_analysis(all_preds, all_labels, label_vocab=LABEL_VOCAB):
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=label_vocab, 
        output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    # Save confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=label_vocab).plot(ax=ax, cmap="Blues")
    plt.tight_layout()

    # Step out of model folder then save into figures folder
    figures_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(os.path.join(figures_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    
    return report, cm