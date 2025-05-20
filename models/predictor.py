from enum import Enum
import xgboost as xgb
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import torch


class ModelType(Enum):
    HUBERT = "hubert"
    XGBOOST = "xgboost"

class Predictor:
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.model = None
        self.feature_extractor = None
    
    def load_model(self):
        if self.model_type == ModelType.HUBERT:
            self.model = HubertForSequenceClassification.from_pretrained('models/hubert/hubert-nemo-emotion')
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('models/hubert/hubert-nemo-emotion')
        elif self.model_type == ModelType.XGBOOST:
            loaded_model = xgb.XGBClassifier()
            loaded_model.load_model('models/xgboost/emotion_xgb_model.json')
            self.model = loaded_model

    def extract_features(self, data):
        if self.model_type == ModelType.HUBERT:
            return self.feature_extractor(
                data,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
        return data

    def predict(self, data):
        if self.model_type == ModelType.HUBERT:
            inputs = self.extract_features(data)
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
            return predictions.numpy()
        else:  # XGBoost
            return self.model.predict(data)
