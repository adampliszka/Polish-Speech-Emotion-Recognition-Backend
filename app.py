import sounddevice as sd
import numpy as np
from models.predictor import Predictor, ModelType
import time
import queue
import threading
import librosa
import torchaudio
import torch
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib as mpl

class EmotionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Emotion Recognition")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2C3E50')  # Dark blue background
        
        # Set theme colors
        self.colors = {
            'bg': '#2C3E50',  # Dark blue
            'fg': '#ECF0F1',  # Light gray
            'accent': '#3498DB',  # Blue
            'success': '#2ECC71',  # Green
            'warning': '#F1C40F',  # Yellow
            'danger': '#E74C3C'  # Red
        }
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure('TFrame', background=self.colors['bg'])
        self.style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['fg'])
        self.style.configure('Title.TLabel', font=('Helvetica', 24, 'bold'))
        self.style.configure('Emotion.TLabel', font=('Helvetica', 20))
        self.style.configure('TCombobox', fieldbackground=self.colors['bg'], background=self.colors['bg'], foreground=self.colors['fg'])
        self.style.configure('Accent.TButton',
            font=('Helvetica', 12, 'bold'),
            background=self.colors['accent'],
            foreground=self.colors['fg'],
            borderwidth=0,
            focusthickness=3,
            focuscolor=self.colors['accent'],
            padding=8
        )
        self.style.map('Accent.TButton',
            background=[('active', '#217dbb'), ('!active', self.colors['accent'])],
            foreground=[('disabled', '#bdc3c7'), ('!disabled', self.colors['fg'])]
        )
        
        # Initialize variables
        self.predictor = None
        self.current_model = None
        self.is_listening = False
        self.stream = None
        
        # Audio parameters
        self.RATE = 16000
        self.CHANNELS = 1
        self.CHUNK_SIZE = int(self.RATE)  # 1 second chunks
        self.BUFFER_SIZE = int(self.RATE)  # 1 second buffer
        
        # Create audio queue and buffer
        self.audio_queue = queue.Queue()
        self.audio_buffer = np.zeros(self.BUFFER_SIZE)
        
        # Create GUI elements
        self.create_widgets()
    
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="20", style='TFrame')
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Title with decorative line
        title_frame = ttk.Frame(main_frame, style='TFrame')
        title_frame.grid(row=0, column=0, columnspan=2, pady=(0, 20), sticky='ew')
        
        title_label = ttk.Label(title_frame, text="Real-time Emotion Recognition", 
                              style='Title.TLabel')
        title_label.pack(pady=(0, 10))
        
        # Decorative line
        canvas = tk.Canvas(title_frame, height=2, bg=self.colors['accent'], highlightthickness=0)
        canvas.pack(fill='x', padx=50)
        
        # Model selection and control frame
        control_frame = ttk.Frame(main_frame, style='TFrame')
        control_frame.grid(row=1, column=0, columnspan=2, pady=(0, 20), sticky='ew')
        
        # Model selection
        model_label = ttk.Label(control_frame, text="Select Model:", style='TLabel')
        model_label.pack(side='left', padx=(0, 10))
        
        self.model_var = tk.StringVar(value='XGBoost')
        model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, 
                                 values=['XGBoost', 'HuBERT'], state='readonly',
                                 width=15)
        model_combo.pack(side='left', padx=(0, 20))
        model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Load button
        self.load_button = ttk.Button(control_frame, text="Load Model", 
                                    command=self.load_model, style='Accent.TButton')
        self.load_button.pack(side='left', padx=(0, 20))
        
        # Start/Stop button
        self.start_button = ttk.Button(control_frame, text="Start Listening", 
                                     command=self.toggle_listening, state='disabled', style='Accent.TButton')
        self.start_button.pack(side='left')
        
        # Current emotion display with background
        emotion_frame = ttk.Frame(main_frame, style='TFrame')
        emotion_frame.grid(row=2, column=0, columnspan=2, pady=20, sticky='ew')
        
        self.emotion_label = ttk.Label(emotion_frame, text="Select and load a model to begin", 
                                     style='Emotion.TLabel')
        self.emotion_label.pack()
        
        # Create bar chart with custom style
        self.fig = Figure(figsize=(10, 5), dpi=100, facecolor=self.colors['bg'])
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, pady=20, sticky='nsew')
        
        # Configure plot style
        self.ax.set_facecolor(self.colors['bg'])
        self.ax.tick_params(colors=self.colors['fg'])
        self.ax.spines['bottom'].set_color(self.colors['fg'])
        self.ax.spines['top'].set_color(self.colors['fg'])
        self.ax.spines['left'].set_color(self.colors['fg'])
        self.ax.spines['right'].set_color(self.colors['fg'])
        
        # Initialize bar chart
        self.emotions = ["anger", "fear", "happiness", "neutral", "sadness", "surprised"]
        self.probs = [0] * len(self.emotions)
        
        # Create bars with custom colors
        bar_colors = [self.colors['danger'], self.colors['warning'], 
                     self.colors['success'], self.colors['accent'],
                     self.colors['warning'], self.colors['success']]
        self.bars = self.ax.bar(self.emotions, self.probs, color=bar_colors)
        
        # Configure plot appearance
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel('Probability', color=self.colors['fg'], fontsize=12)
        self.ax.set_title('Emotion Probabilities', color=self.colors['fg'], fontsize=14, pad=20)
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha='right', color=self.colors['fg'])
        plt.setp(self.ax.get_yticklabels(), color=self.colors['fg'])
        
        # Add grid
        self.ax.grid(True, linestyle='--', alpha=0.3, color=self.colors['fg'])
        
        # Adjust layout
        self.fig.tight_layout()
        
        # Add status bar
        status_frame = ttk.Frame(main_frame, style='TFrame')
        status_frame.grid(row=4, column=0, columnspan=2, sticky='ew', pady=(20, 0))
        
        self.status_label = ttk.Label(status_frame, 
                                    text="Select a model and click 'Load Model' to begin", 
                                    style='TLabel')
        self.status_label.pack(side='left')
    
    def load_model(self):
        model_name = self.model_var.get()
        model_type = ModelType.XGBOOST if model_name == 'XGBoost' else ModelType.HUBERT
        
        self.status_label.config(text=f"Loading {model_name} model...")
        self.root.update()
        
        try:
            if self.predictor is not None:
                self.predictor = None  # Clean up previous model
            self.predictor = Predictor(model_type)
            self.predictor.load_model()
            self.current_model = model_type
            
            self.status_label.config(text=f"{model_name} model loaded successfully")
            self.start_button.config(state='normal')
            self.emotion_label.config(text="Click 'Start Listening' to begin")
        except Exception as e:
            self.status_label.config(text=f"Error loading model: {str(e)}")
            self.start_button.config(state='disabled')
    
    def toggle_listening(self):
        if not self.is_listening:
            # Start listening
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=self.CHANNELS,
                samplerate=self.RATE,
                blocksize=self.CHUNK_SIZE,
                dtype=np.float32
            )
            self.stream.start()
            self.is_listening = True
            self.start_button.config(text="Stop Listening")
            self.status_label.config(text=f"Listening... ({self.model_var.get()})")
            self.process_audio()
        else:
            # Stop listening
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            self.is_listening = False
            self.start_button.config(text="Start Listening")
            self.status_label.config(text=f"Stopped. Model: {self.model_var.get()}")
            self.emotion_label.config(text="Click 'Start Listening' to begin")
    
    def on_model_change(self, event):
        if self.is_listening:
            self.toggle_listening()  # Stop current listening
        self.start_button.config(state='disabled')
        self.status_label.config(text="Click 'Load Model' to load the selected model")
        self.emotion_label.config(text="Select and load a model to begin")
    
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        self.audio_queue.put(indata.copy())
    
    def extract_features(self, audio_data, sr=16000):
        if self.current_model == ModelType.XGBOOST:
            # Convert to torch tensor
            waveform = torch.tensor(audio_data).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            
            # Extract MFCCs using torchaudio (matching training pipeline)
            mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=40)
            mfcc = mfcc_transform(waveform)
            features = mfcc.mean(dim=2).squeeze().numpy()
            
            return features.reshape(1, -1)  # Reshape for XGBoost
        else:
            return audio_data  # HuBERT model handles feature extraction internally
    
    def update_plot(self, probabilities):
        # Update bar heights with smooth animation
        for bar, prob in zip(self.bars, probabilities):
            current_height = bar.get_height()
            new_height = prob
            # Smooth transition
            bar.set_height(current_height + (new_height - current_height) * 0.3)
        
        # Update canvas
        self.canvas.draw()
    
    def process_audio(self):
        if not self.is_listening:
            return
            
        try:
            # Get new audio data
            audio_data = self.audio_queue.get_nowait()
            
            # Update buffer
            self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
            self.audio_buffer[-len(audio_data):] = audio_data.flatten()
            
            # Normalize audio data
            if np.max(np.abs(self.audio_buffer)) > 1.0:
                self.audio_buffer = self.audio_buffer / np.max(np.abs(self.audio_buffer))
            
            # Extract features and get prediction
            features = self.extract_features(self.audio_buffer)
            predictions, probabilities = self.predictor.predict_with_probs(features)
            
            if predictions is not None and len(predictions) > 0:
                # Update emotion label with color
                predicted_emotion = self.emotions[predictions[0]]
                self.emotion_label.config(
                    text=f"Current Emotion: {predicted_emotion}",
                    foreground=self.colors['success']
                )
                
                # Update probability plot
                self.update_plot(probabilities[0])
                
                # Update status
                self.status_label.config(text=f"Processing audio... ({self.model_var.get()})")
            
        except queue.Empty:
            pass
        
        # Schedule next update if still listening
        if self.is_listening:
            self.root.after(50, self.process_audio)
    
    def cleanup(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

def main():
    root = tk.Tk()
    app = EmotionGUI(root)
    
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
