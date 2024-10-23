import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LibriTTSDataset(Dataset):
    """
    A PyTorch dataset for the LibriTTS dataset.
    """

    def __init__(self, root_dir, subset="LibriTTS\\train-clean-100"):
        """
        Initialize the dataset.

        Args:
            root_dir (str): The root directory of the dataset.
            subset (str): The subset of the dataset to use. Defaults to "train-clean-100".
        """
        self.root_dir = os.path.join(root_dir, subset)
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        """
        Load the metadata for the dataset.

        Returns:
            list: A list of tuples, where each tuple contains the path to a
                WAV file and the path to its corresponding text file.
        """
        metadata = []
        for speaker_id in os.listdir(self.root_dir):
            speaker_dir = os.path.join(self.root_dir, speaker_id)
            for chapter_id in os.listdir(speaker_dir):
                chapter_dir = os.path.join(speaker_dir, chapter_id)
                for file in os.listdir(chapter_dir):
                    if file.endswith(".wav"):
                        wav_path = os.path.join(chapter_dir, file)
                        txt_path = wav_path.replace(".wav", ".normalized.txt")
                        if os.path.exists(txt_path):
                            metadata.append((wav_path, txt_path))
        return metadata

    def __len__(self):
        """
        Return the size of the dataset.

        Returns:
            int: The size of the dataset.
        """
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the waveform, sample rate, and text of the sample.
        """
        wav_path, txt_path = self.metadata[idx]
        print(f"Loading {wav_path}")
        waveform, sample_rate = torchaudio.load(wav_path, format="wav")
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return waveform, sample_rate, text

dataset = LibriTTSDataset(root_dir="./data")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

print(f"Dataset size: {len(dataset)}")
waveform, sample_rate, text = dataset[0]
print(f"Sample rate: {sample_rate}")
print(f"Waveform shape: {waveform.shape}")
print(f"Text sample: {text}")
