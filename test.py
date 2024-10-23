import torchaudio
import torch

print(torch.__version__)
# List available audio backends
print(torchaudio.__version__)
print("Available audio backends:", torchaudio.list_audio_backends())

# Try loading a sample audio file (make sure the path is correct)
try:
    waveform, sample_rate = torchaudio.load("path/to/your/audio.wav")
    print("Waveform shape:", waveform.shape)
    print("Sample rate:", sample_rate)
except Exception as e:
    print("Error loading audio:", e)
