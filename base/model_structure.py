from torch import nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, output_size):
        super(LSTMClassifier, self).__init__()

        embed_size = 20
        hidden_size = 250  # idk

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, num_layers=2, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch_size, sequence_length, embed_size)
        lstm_out, (h_n, _) =  self.lstm(x)  # Shape: (batch_size, sequence_length, hidden_size)
        # h_n = h_n[-1]  # Get the last hidden state
        h_n = h_n[-2:].transpose(0, 1).contiguous().view(x.size(0), -1)
        out = self.dropout(self.fc1(h_n))  # Shape: (batch_size, output_size)
        return self.softmax(out)