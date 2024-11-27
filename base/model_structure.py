from torch import nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, output_size)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.embedding(x))  # Shape: (batch_size, sequence_length, embed_size)
        lstm_out, (h_n, _) =  self.lstm(x)  # Shape: (batch_size, sequence_length, hidden_size)
        h_n = h_n[-1]  # Get the last hidden state
        out = self.dropout(self.fc1(h_n))  # Shape: (batch_size, output_size)
        out = self.dropout(self.fc2(self.sigmoid(out)))
        return self.sigmoid(out)