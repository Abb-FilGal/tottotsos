import soundfile as sf
import torchaudio

print(sf.__version__)  # Check if soundfile is installed
print(torchaudio.__version__)  # Check torchaudio version
print("Available audio backends:", torchaudio.list_audio_backends())
