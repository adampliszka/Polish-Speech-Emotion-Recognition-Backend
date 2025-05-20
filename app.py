import sounddevice as sd
import numpy as np
from models.predictor import Predictor, ModelType

def record_audio(seconds=5):
    # Audio recording parameters
    RATE = 16000
    CHANNELS = 1

    print("Available audio input devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # Only show input devices
            print(f"Device {i}: {device['name']}")

    print(f"* Recording for {seconds} seconds...")
    
    # Record audio
    recording = sd.rec(
        int(seconds * RATE),
        samplerate=RATE,
        channels=CHANNELS,
        dtype=np.float32
    )
    sd.wait()  # Wait until recording is finished
    
    print("* Done recording")
    
    # Convert to mono if stereo
    if recording.shape[1] == 2:
        audio_data = np.mean(recording, axis=1)
    else:
        audio_data = recording.flatten()
        
    return audio_data, RATE

def main():
    # Initialize predictor with HuBERT model
    predictor = Predictor(ModelType.HUBERT)
    predictor.load_model()
    
    print("Emotion Recognition System")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            input("Press Enter to start recording...")
            
            # Record audio
            audio_data, sample_rate = record_audio(seconds=5)
            
            # Get prediction
            prediction = predictor.predict(audio_data)
            
            # Map prediction to emotion (assuming 0: neutral, 1: happy, 2: sad, 3: angry)
            emotions = ["neutral", "happy", "sad", "angry"]
            predicted_emotion = emotions[prediction[0]]
            
            print(f"Predicted emotion: {predicted_emotion}")
            print("-" * 50)
            
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
