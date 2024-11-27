import json, configparser
import sys
import onnxruntime
import base.model_tools as model_tools
from flask import Flask, jsonify, request

config = configparser.ConfigParser()
config.read("./MODEL_CONFIG.ini")

device = config.get("Server", "device") # useless

app = Flask(__name__)

recognition_model_session = onnxruntime.InferenceSession(config.get("Recognition", "save_path"))
recognition_vocab = json.load(open(config.get("Recognition", "vocab_path"), encoding="utf-8"))

@app.route('/recognition')
def recognition():
    text = request.args.get('text', default = "", type = str)
    print(text)
    if not text:
        return jsonify({"result": None, "error": "Param 'text' is null"})
    answer, confidence = model_tools.simple_forward_onnx(recognition_model_session, recognition_vocab, text.lower())
    confidence = float(confidence)

    return jsonify({'result': {
        "answer": answer, "confidence": confidence
    }, 'error': None})

if __name__ == '__main__':
    app.run(debug=False, port=int(sys.argv[1]))
