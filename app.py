from flask import Flask, request, jsonify
from vosk import Model, KaldiRecognizer
import wave
import json
import os
import subprocess
import zipfile
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_ZIP = "vosk-model-small-en-us-0.15.zip"
MODEL_PATH = "vosk-model-small-en-us-0.15"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading Vosk model...")
        os.system(f"wget https://alphacephei.com/vosk/models/{MODEL_ZIP} -O {MODEL_ZIP}")
        print("Unzipping model...")
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(MODEL_ZIP)
        print("Model ready!")

download_model()

# Now load the model from its actual folder name
model = Model(MODEL_PATH)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files["file"]
    input_path = "input_audio"
    uploaded_file.save(input_path)

    # Convert to WAV 16kHz mono
    wav_path = "converted.wav"
    subprocess.run([
        "ffmpeg", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-f", "wav", wav_path,
        "-y"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())

    result_text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result_text += json.loads(rec.Result())["text"] + " "
    result_text += json.loads(rec.FinalResult())["text"]

    # Cleanup
    os.remove(input_path)
    os.remove(wav_path)

    return jsonify({"transcription": result_text.strip()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
