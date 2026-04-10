import torch

LABEL_VOCAB = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]

def evaluate(model, iterator, criterion):
    model.eval()
    device = next(model.parameters()).device
    epoch_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            specs, labels = batch
            specs, labels = specs.to(device), labels.to(device)

            logits = model(specs)

            loss = criterion(logits, labels)

            epoch_loss += loss.item()

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    accuracy = correct / total if total > 0 else 0.0
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    return epoch_loss / len(iterator), accuracy, all_preds, all_labels
