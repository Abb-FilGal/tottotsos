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

    melspec = librosa.feature.melspectrogram(audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    melspec = np.log(melspec + 1e-8)  # Apply logarithmic compression

    return melspec, audio

