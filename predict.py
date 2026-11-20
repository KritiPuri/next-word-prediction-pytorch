import json
from pathlib import Path

import torch

from model import NextWordLSTM
from tokenizer import WordTokenizer

ARTIFACTS = Path("artifacts")

def load_model_and_tokenizer(artifacts_dir=ARTIFACTS):
    artifacts_dir = Path(artifacts_dir)
    checkpoint = torch.load(artifacts_dir / "next_word_lstm.pt", map_location="cpu", weights_only=False)
    tokenizer_data = json.loads((artifacts_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer = WordTokenizer.from_dict(tokenizer_data)
    model = NextWordLSTM(
        checkpoint["vocab_size"],
        checkpoint["embedding_dim"],
        checkpoint["hidden_dim1"],
        checkpoint["hidden_dim2"],
        checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, checkpoint["max_sequence_len"]

def predict_next_word(text, model, tokenizer, max_sequence_len):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = sequence[-(max_sequence_len - 1):]
    sequence = [0] * (max_sequence_len - 1 - len(sequence)) + sequence
    x = torch.tensor([sequence], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
        index = int(logits.argmax(dim=1).item())
    return tokenizer.index_word.get(index, "<unknown>")

if __name__ == "__main__":
    model, tokenizer, max_len = load_model_and_tokenizer()
    sample = input("Enter a sequence: ")
    print("Next word:", predict_next_word(sample, model, tokenizer, max_len))
