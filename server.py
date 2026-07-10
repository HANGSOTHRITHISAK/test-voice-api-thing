import os
import sys
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder=".", static_url_path="")

# Load model once at startup
model = None


def load_model():
    from faster_whisper import WhisperModel
    return WhisperModel(
        "Tnaot/whisper-large-v3-khmer-ct2",
        device="cuda",
        compute_type="int8_float16",
    )


def get_model():
    global model
    if model is None:
        model = load_model()
    return model


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio"]
    ext = Path(file.filename).suffix if file.filename else ".wav"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        m = get_model()
        segments, info = m.transcribe(
            tmp_path,
            language="km",
            task="transcribe",
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return jsonify({
            "text": text,
            "detected_language": info.language,
            "duration": info.duration,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    print("Loading Khmer Whisper model (first request will trigger download if needed)...")
    get_model()
    print("Model loaded. Starting server on http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)
