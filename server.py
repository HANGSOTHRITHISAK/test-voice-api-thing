import os
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from google.cloud import speech

app = Flask(__name__, static_folder=".", static_url_path="")

client = speech.SpeechClient()


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
        with open(tmp_path, "rb") as f:
            content = f.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            language_code="km-KH",
            enable_automatic_punctuation=True,
            model="latest_long",
        )

        response = client.recognize(config=config, audio=audio)

        text = " ".join(
            result.alternatives[0].transcript for result in response.results
        )

        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on 0.0.0.0:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)
