import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import NextWordLSTM
from tokenizer import WordTokenizer, tokenize

def make_sequences(text, tokenizer):
    sequences = []
    for line in text.splitlines():
        tokens = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(tokens)):
            sequences.append(tokens[:i + 1])
    if not sequences:
        raise ValueError("No training sequences were generated. Check hamlet.txt.")
    max_len = max(len(seq) for seq in sequences)
    x, y = [], []
    for seq in sequences:
        padded = [0] * (max_len - len(seq)) + seq
        x.append(padded[:-1])
        y.append(padded[-1])
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long), max_len

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", default="hamlet.txt")
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    text = Path(args.data).read_text(encoding="utf-8").lower()
    tokenizer = WordTokenizer()
    tokenizer.fit_on_texts([text])
    x, y, max_sequence_len = make_sequences(text, tokenizer)

    indices = torch.randperm(len(x))
    split = int(len(x) * 0.8)
    train_idx, val_idx = indices[:split], indices[split:]
    train_ds = TensorDataset(x[train_idx], y[train_idx])
    val_ds = TensorDataset(x[val_idx], y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = NextWordLSTM(len(tokenizer.word_index) + 1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(yb)
        train_loss, train_acc = total_loss / total, correct / total

        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * len(yb)
                val_correct += (logits.argmax(dim=1) == yb).sum().item()
                val_total += len(yb)
        val_loss, val_acc = val_loss_sum / val_total, val_correct / val_total
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_size": len(tokenizer.word_index) + 1,
        "max_sequence_len": max_sequence_len,
        "embedding_dim": 100,
        "hidden_dim1": 150,
        "hidden_dim2": 100,
        "dropout": 0.2,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, output_dir / "next_word_lstm.pt")
    (output_dir / "tokenizer.json").write_text(json.dumps(tokenizer.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model and tokenizer in: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
