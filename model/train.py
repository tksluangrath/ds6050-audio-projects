import torch


def train(model, iterator, optimizer, criterion, clip, scheduler=None):
    model.train()
    device = next(model.parameters()).device
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        specs, labels = batch
        specs, labels = specs.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(specs)

        loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    if scheduler is not None:
        scheduler.step()

    return epoch_loss / len(iterator)
