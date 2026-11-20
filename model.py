import torch
from torch import nn

class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim1=150, hidden_dim2=100, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.fc = nn.Linear(hidden_dim2, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, (hidden, _) = self.lstm2(x)
        return self.fc(hidden[-1])
