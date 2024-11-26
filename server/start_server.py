import json
import base.model as model_meta
from base.model import LSTMClassifier
from flask import Flask, jsonify, request
from torch import load as model_load

device = "cpu" # or "cuda"

app = Flask(__name__)

model = model_load("../AsyaRecModel").to(device)
vocab = json.load(open("../vocab.json", encoding="utf-8"))

@app.route('/recognition')
def recognition():
    text = request.args.get('text', default = "", type = str)
    if not text:
        return jsonify({"result": None, "error": "Param 'text' is null"})

    answer, confidence = model_meta.simple_forward(model, vocab, text, device=device)
    return jsonify({'result': {
        "answer": answer, "confidence": confidence
    }, 'error': None})

if __name__ == '__main__':
    app.run(debug=True)