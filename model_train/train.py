import torch
import nltk
import json
nltk.download('punkt_tab')
import torch.nn as nn
import torch.optim as optim
import base.model as model_meta
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DEVICE = "cuda:0" # or "cpu"
TRAIN_EPOCHS = 50

SAVE_PATH = "AsyaRecModel"
VOCAB_PATH = "vocab.json"
DATASET_PATH = "dataset.json"

dataset = json.load(open(DATASET_PATH, "r", encoding="utf-8"))

total_text = []
labels = []

for label_name, answers in dataset.items():
    for i in answers:
        total_text.append(i
          .replace(".", "")
          .replace(",", "")
          .replace("?", "")
        )
        labels.append(model_meta.labels_matrix[model_meta.labels_text.index(label_name)])

tokenized_texts = [text.lower().split(" ") for text in total_text]

# Build vocabulary
vocab = Counter(word for sentence in tokenized_texts for word in sentence)
vocab = {word: i+1 for i, (word, count) in enumerate(vocab.items())} # start from 1 for the embedding
vocab_reversed =  {i: word for word, i in vocab.items()}

# Save vocab
f = open(VOCAB_PATH, "w", encoding="utf-8")
f.write(json.dumps(vocab, ensure_ascii=False))
f.close()

# Convert texts to sequences of indices
sequences = [[vocab[word] for word in sentence] for sentence in tokenized_texts]

# Pad sequences to the same length
max_length = max(len(seq) for seq in sequences)
padded_sequences = [seq + [0] * (max_length - len(seq)) for seq in sequences]

# Convert to tensors
X = torch.tensor(padded_sequences, dtype=torch.long)
y = labels

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Create DataLoader
train_dataset = TextDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataset = TextDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model parameters
vocab_size = len(vocab) + 1
embed_size = 100
hidden_size = 1024 # idk
output_size = len(model_meta.labels_text)

model = model_meta.LSTMClassifier(vocab_size, embed_size, hidden_size, output_size).to(DEVICE)
criterion = nn.BCELoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(TRAIN_EPOCHS):
    for texts, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(texts.to(DEVICE))
        loss = criterion(outputs, labels.to(DEVICE))
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

model.eval()

# Test
with torch.no_grad():
    for texts, labels in test_loader:
        scores = model(texts.to(DEVICE))
        print(model_meta.decode(vocab_reversed, texts.squeeze()), model_meta.get_best(scores.squeeze()))

while True:
    i = input("Save this model? Y/N").lower()
    if i == "y":
        torch.save(model, SAVE_PATH)
    elif i == "n":
        break