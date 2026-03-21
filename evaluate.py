import torch


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            specs, labels = batch

            logits = model(specs)

            loss = criterion(logits, labels)

            epoch_loss += loss.item()

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return epoch_loss / len(iterator), accuracy
