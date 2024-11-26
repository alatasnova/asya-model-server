import configparser, json
import re
from torch import nn, argmax, eye, tensor

config = configparser.ConfigParser()
config.read("./MODEL_CONFIG.ini")

labels_text = json.loads(config.get("Recognition", "labels"))

labels_matrix = eye(len(labels_text))

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

def tokenize(text):
    # return list(text)
    return re.split(r"(\s+)", text)

def get_best(scores):
    best_idx = argmax(scores)
    return labels_text[best_idx], scores[best_idx].item()

def encode(vocab, text):
    tokenized_text = tokenize(text.lower())

    sequence = []
    for word in tokenized_text:
        item = vocab.get(word)
        if item is None:
            item = 0
        sequence.append(item)
    return tensor([sequence])

def decode(vocab_reversed, x):
    sequence = []
    for idx in x:
        if idx == 0:
            break
        word = vocab_reversed.get(idx.item())
        if word is None:
            word = "?"
        sequence.append(word)
    return "".join(sequence)

def simple_forward(model, vocab, text, device="cpu"):
    scores = model(encode(vocab, text).to(device))
    return get_best(scores.squeeze())
