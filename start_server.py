import sys
import json
import onnxruntime
import base.model_tools as model_tools

from flask import Flask, jsonify, request
from base.config import SERVER_DEVICE, RECOGNITION_SAVE_PATH, RECOGNITION_VOCAB_PATH

app = Flask(__name__)

recognition_model_session = onnxruntime.InferenceSession(RECOGNITION_SAVE_PATH)
recognition_vocab = json.load(open(RECOGNITION_VOCAB_PATH, encoding="utf-8"))

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

def try_get(arr, i):
    if i >= len(arr):
        return None
    if i < 0:
        return None
    return arr[i]

if __name__ == '__main__':
    inp_port = try_get(sys.argv, 1)
    inp_host = try_get(sys.argv, 2)

    if inp_port is None:
        inp_port = input("Enter port (default 5000):")
    if inp_host is None:
        inp_host = "127.0.0.1"

    if inp_port.isdigit():
        inp_port = int(inp_port)
    else:
        raise ValueError("Error: Expected port argument to be a number")


    app.run(debug=False, port=inp_port, host=inp_host)
