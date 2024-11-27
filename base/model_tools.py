import json
import configparser
import re
import numpy as np

config = configparser.ConfigParser()
config.read("./MODEL_CONFIG.ini")

labels_text = json.loads(config.get("Recognition", "labels"))

labels_matrix = np.eye(len(labels_text))

def tokenize(text):
    # return list(text)
    return re.split(r"(\s+)", text)

def get_best(scores):
    best_idx = np.argmax(scores)
    return labels_text[best_idx], scores[0][best_idx]

def encode(vocab, text):
    tokenized_text = tokenize(text.lower())

    sequence = []
    for word in tokenized_text:
        item = vocab.get(word)
        if item is None:
            item = 0
        sequence.append(item)

    return np.array([sequence])

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

def simple_forward(model, vocab, text):
    scores = model(encode(vocab, text))
    return get_best(scores.squeeze())

def simple_forward_onnx(ort_session, vocab, text):
    ort_inputs = {"text": encode(vocab, text)}
    ort_outs = ort_session.run(None, ort_inputs)
    return get_best(ort_outs[0])