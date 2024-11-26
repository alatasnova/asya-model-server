import json, configparser
import sys
import base.model_meta as model_meta
from base.model_meta import LSTMClassifier
from flask import Flask, jsonify, request
from torch import load

config = configparser.ConfigParser()
config.read("./MODEL_CONFIG.ini")

device = config.get("Server", "device") # or "cuda"

app = Flask(__name__)

recognition_model = load(config.get("Recognition", "save_path")).to(device)
recognition_vocab = json.load(open(config.get("Recognition", "vocab_path"), encoding="utf-8"))

@app.route('/recognition')
def recognition():
    text = request.args.get('text', default = "", type = str)
    print(text)
    if not text:
        return jsonify({"result": None, "error": "Param 'text' is null"})

    answer, confidence = model_meta.simple_forward(recognition_model, recognition_vocab, text.lower(), device=device)
    return jsonify({'result': {
        "answer": answer, "confidence": confidence
    }, 'error': None})

if __name__ == '__main__':
    app.run(debug=False, port=int(sys.argv[1]))
