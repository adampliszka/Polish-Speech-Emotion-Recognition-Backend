from enum import Enum
import xgboost as xgb
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import torchaudio
import numpy as np


class ModelType(Enum):
    HUBERT = "hubert"
    XGBOOST = "xgboost"

class Predictor:
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.model = None
        self.feature_extractor = None
        # Initialize MFCC transform for XGBoost (matching training exactly)
        if model_type == ModelType.XGBOOST:
            self.mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=16000, 
                n_mfcc=40
            )
    
    def load_model(self):
        if self.model_type == ModelType.HUBERT:
            self.model = HubertForSequenceClassification.from_pretrained('models/hubert/hubert-nemo-emotion')
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('models/hubert/hubert-nemo-emotion')
        elif self.model_type == ModelType.XGBOOST:
            loaded_model = xgb.XGBClassifier()
            loaded_model.load_model('models/xgboost/emotion_xgb_model.json')
            self.model = loaded_model

    def extract_features(self, data, sample_rate=16000):
        if self.model_type == ModelType.HUBERT:
            return self.feature_extractor(
                data,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
        else:  # XGBoost
            # Convert to tensor if it's not already
            if not isinstance(data, torch.Tensor):
                waveform = torch.tensor(data).float()
            else:
                waveform = data.float()
            
            # Ensure proper shape (add channel dimension if needed)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            
            # IMPORTANT: Resampling can cause prediction errors if sample_rate is incorrect
            # The model was trained on 16kHz audio. If unsure about sample rate, use 16000.
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Handle very short audio by padding to minimum length
            min_length = 16000  # 1 second at 16kHz (more robust than 400 samples)
            if waveform.shape[1] < min_length:
                padding_needed = min_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
            
            # Handle very long audio by taking first 5 seconds (optional)
            max_length = 16000 * 5  # 5 seconds
            if waveform.shape[1] > max_length:
                waveform = waveform[:, :max_length]
            
            # Extract MFCC features (matching training pipeline exactly)
            mfcc = self.mfcc_transform(waveform)
            
            # Take mean across time dimension and squeeze to get feature vector
            features = mfcc.mean(dim=2).squeeze()
            
            # Convert to numpy for XGBoost
            return features.numpy()

    def predict(self, data, sample_rate=16000):
        if self.model_type == ModelType.HUBERT:
            inputs = self.extract_features(data)
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
            return predictions.numpy()
        else:  # XGBoost
            features = self.extract_features(data, sample_rate)
            # Ensure features are 2D for XGBoost (reshape if single sample)
            if features.ndim == 1:
                features = features.reshape(1, -1)
            return self.model.predict(features)
            
    def predict_with_probs(self, data, sample_rate=16000):
        if self.model_type == ModelType.HUBERT:
            inputs = self.extract_features(data)
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                probabilities = torch.softmax(outputs.logits, dim=-1)
            return predictions.numpy(), probabilities.numpy()
        else:  # XGBoost
            features = self.extract_features(data, sample_rate)
            # Ensure features are 2D for XGBoost (reshape if single sample)
            if features.ndim == 1:
                features = features.reshape(1, -1)
            predictions = self.model.predict(features)
            probabilities = self.model.predict_proba(features)
            return predictions, probabilities
