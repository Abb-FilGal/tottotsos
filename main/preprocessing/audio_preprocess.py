import os
import librosa
import numpy as np

def extract_mel_spectrogram(file_path, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=80):
    try:
        audio, _ = librosa.load(file_path, sr=sample_rate)
        audio = audio / np.max(np.abs(audio))
        
        melspec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        melspec = np.log(melspec + 1e-8)  # Apply logarithmic compression
        
        return melspec, audio
    except Exception as e:
        print(f"Error extracting mel spectrogram from {file_path}: {str(e)}")
        return None, None

def normalize(melspec):
    if melspec is None:
        return None
    mean = np.mean(melspec)
    std = np.std(melspec)
    return (melspec - mean) / std

def save_training_data(melspec, audio, output_dir, file_name):
    if melspec is None or audio is None:
        print(f"Cannot save training data for {file_name}: melspec or audio is None")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    full_path = os.path.join(output_dir, file_name + '.npz')
    np.savez(full_path, melspec=melspec, audio=audio)

def preprocess(file_path, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=80):
    melspec, audio = extract_mel_spectrogram(file_path, sample_rate, n_fft, hop_length, n_mels)
    if melspec is None or audio is None:
        return None, None

    melspec = normalize(melspec)
    
    map_name = file_path.replace(
        os.path.normpath("./data/LibriTTS/dev-clean"),
        os.path.normpath("./data/processed")
    )
    output_dir = os.path.join("./data", os.path.dirname(map_name), "training_data")
    file_name = os.path.basename(file_path).replace(".wav", "")
    
    save_training_data(melspec, audio, output_dir=output_dir, file_name=file_name)
    
    return melspec, audio

if __name__ == "__main__":
    # Example usage
    result = preprocess("./data/LibriTTS/dev-clean/84/121123/84_121123_000001_000000.wav")
    if result[0] is not None and result[1] is not None:
        print("Preprocessing successful")
    else:
        print("Preprocessing failed")