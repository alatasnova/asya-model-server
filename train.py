import torch
from importlib import reload
import nltk
nltk.download('punkt_tab')
import onnxruntime
import json
import base.model_structure as model_structure
import base.model_tools as model_tools

import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split

from base.model_tools import tokenize
from base.config import TRAIN_DEVICE, TRAIN_NUM_EPOCHS, RECOGNITION_SAVE_PATH, RECOGNITION_VOCAB_PATH, RECOGNITION_DATASET_PATH, RECOGNITION_LABELS

# FOR DEBUG
model_structure = reload(model_structure)
model_tools = reload(model_tools)

dataset = json.load(open(RECOGNITION_DATASET_PATH, "r", encoding="utf-8"))

total_text = []
labels = []

for label_name, answers in dataset.items():
    for i in answers:
        total_text.append(i
                          .replace(".", "")
                          .replace(",", "")
                          .replace("?", "")
                          )
        labels.append(torch.from_numpy(model_tools.labels_matrix[RECOGNITION_LABELS.index(label_name)]).float())

tokenized_texts = [tokenize(text.lower()) for text in total_text]

# Build vocabulary
vocab = Counter(word for sentence in tokenized_texts for word in sentence)
vocab = {word: i + 1 for i, (word, count) in enumerate(vocab.items())}  # start from 1 for the embedding
vocab_reversed = {i: word for word, i in vocab.items()}

# Save vocab
f = open(RECOGNITION_VOCAB_PATH, "w", encoding="utf-8")
f.write(json.dumps(vocab, ensure_ascii=False))
f.close()

# Convert texts to sequences of indices
sequences = [[vocab[word] for word in sentence] for sentence in tokenized_texts]

# Pad sequences to the same length
max_length = max(len(seq) for seq in sequences)
padded_sequences = [seq + [0] * (max_length - len(seq)) for seq in sequences]

# Convert to tensors
X = torch.tensor(padded_sequences).int()
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.004, random_state=42)

# Create DataLoader
train_dataset = TextDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = TextDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model parameters
vocab_size = len(vocab) + 1
output_size = len(RECOGNITION_LABELS)

model = model_structure.LSTMClassifier(vocab_size, output_size).to(TRAIN_DEVICE)
criterion = nn.BCELoss().to(TRAIN_DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(TRAIN_NUM_EPOCHS):
    for texts, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(texts.to(TRAIN_DEVICE))
        loss = criterion(outputs, labels.to(TRAIN_DEVICE))
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

model.eval()
# Test
with torch.no_grad():
    for texts, labels in test_loader:
        scores = model(texts.to(TRAIN_DEVICE))
        print(model_tools.decode(vocab_reversed, texts.squeeze().numpy()), model_tools.get_best(scores.squeeze().cpu().numpy()))


def save_model():
    model.to("cpu")
    x = torch.from_numpy(model_tools.encode(vocab, "ася включи музыку"))
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,               # model being run
                      (x,),                         # model input (or a tuple for multiple inputs)
                      RECOGNITION_SAVE_PATH,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      input_names = ['text'],   # the model's input names
                      output_names = ['labels'], # the model's output names
                      dynamic_axes={
                          'text' : {0: 'batch_size', 1 : 'sequence_length'},
                          'labels': {0: 'batch_size', 1 : 'sequence_length'}
                      })

    ort_session = onnxruntime.InferenceSession(RECOGNITION_SAVE_PATH)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("ONNX Model test success")

while True:
    i = input("Save this model? Y/N").lower()
    if i == "y":
        save_model()
        break
    elif i == "n":
        break
