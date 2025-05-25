from flask import Flask, request, jsonify
from flask_cors import CORS
from models.predictor import Predictor, ModelType
import numpy as np
import torchaudio
import torch
import threading

app = Flask(__name__)
CORS(app)

class ModelManager:
    def __init__(self):
        self.predictor = None
        self.current_model = None
        self.lock = threading.Lock()

    def load_model(self, model_type):
        with self.lock:
            if self.predictor is not None:
                self.predictor = None
            self.predictor = Predictor(model_type)
            self.predictor.load_model()
            self.current_model = model_type

    def predict(self, features):
        with self.lock:
            return self.predictor.predict_with_probs(features)

model_manager = ModelManager()

@app.route('/load_model', methods=['POST'])
def load_model():
    data = request.json
    model_name = data.get('model_name')

    if not model_name:
        return jsonify({"error": "Model name is required"}), 400

    try:
        model_type = ModelType.XGBOOST if model_name == 'XGBoost' else ModelType.HUBERT
        model_manager.load_model(model_type)
        return jsonify({"message": f"{model_name} model loaded successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500


@app.route('/predict', methods=['POST'])
def predict():
    if not model_manager.predictor:
        return jsonify({"error": "Model not loaded"}), 400

    try:
        audio_data = request.json.get('audio_data')
        sr = request.json.get('sample_rate', 16000)

        if not isinstance(audio_data, list) or not all(isinstance(x, (int, float)) for x in audio_data):
            return jsonify({"error": "Invalid audio data"}), 400

        if not isinstance(sr, int) or sr <= 0:
            return jsonify({"error": "Invalid sample rate"}), 400

        audio_data = np.array(audio_data, dtype=np.float32)
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        if model_manager.current_model == ModelType.XGBOOST:
            waveform = torch.tensor(audio_data, dtype=torch.float32)  # Ensure tensor is float32
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)

            # Dynamically adjust n_fft based on input size
            input_size = waveform.size(1)
            n_fft = min(400, input_size)

            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sr,
                n_mfcc=40,
                melkwargs={
                    "n_fft": n_fft,
                    "hop_length": 160,
                    "win_length": n_fft,
                    "n_mels": 40,
                    "center": False
                }
            )
            mfcc = mfcc_transform(waveform)
            features = mfcc.mean(dim=2).squeeze().numpy().reshape(1, -1)
        else:
            features = audio_data.astype(np.float32).reshape(1, -1)

        predictions, probabilities = model_manager.predict(features)
        emotions = ["anger", "fear", "happiness", "neutral", "sadness", "surprised"]
        return jsonify({
            "predicted_emotion": emotions[predictions[0]],
            "probabilities": probabilities[0].tolist()
        }), 200
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)