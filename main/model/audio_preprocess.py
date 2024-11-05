import librosa
import numpy as np
import os


import librosa
import numpy as np

def extract_mel_spectrogram(file_path, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=80):
    """
    Extract a mel-scaled spectrogram from an audio file.

    Args:
        file_path (str): The path to the audio file.
        sample_rate (int, optional): The sample rate of the audio file. Defaults to 24000.
        n_fft (int, optional): The number of frequency bins in the FFT. Defaults to 1024.
        hop_length (int, optional): The number of samples between successive frames. Defaults to 256.
        n_mels (int, optional): The number of mel bins. Defaults to 80.

    Returns:
        tuple: A tuple containing the mel-scaled spectrogram and the original audio signal.
    """
    audio, _ = librosa.load(file_path, sr=sample_rate)
    audio = audio / np.max(np.abs(audio))

    # Pass the audio signal as a keyword argument
    melspec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    melspec = np.log(melspec + 1e-8)  # Apply logarithmic compression

    return melspec, audio


def normalize(melspec):
    """
    Normalize the mel-scaled spectrogram.

    Args:
        melspec (numpy.ndarray): The mel-scaled spectrogram to normalize.

    Returns:
        numpy.ndarray: The normalized mel-scaled spectrogram.
    """
    mean = np.mean(melspec)
    std = np.std(melspec)
    return (melspec - mean) / std

def save_training_data(melspec, audio, output_dir, file_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(os.path.join(output_dir, file_name)):
        print(f"Skipping {file_name} because it already exists.")
        return
    np.savez(os.path.join(output_dir, file_name), melspec=melspec, audio=audio)


def preprocess(file_path, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=80):
    melspec, audio = extract_mel_spectrogram(file_path, sample_rate, n_fft, hop_length, n_mels)
    output_dir = os.path.join("./data", os.path.dirname(file_path), "training_data")
    file_name = os.path.basename(file_path).replace(".wav", "")
    save_training_data(melspec, audio, output_dir=output_dir, file_name=file_name)