import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np

os.system('cls' if os.name == 'nt' else 'clear')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torchaudio.datasets.LIBRITTS(root="./data", url="dev-clean", download=True)

class LibriTTSDataset(Dataset):
    """
    A PyTorch dataset for the LibriTTS dataset.
    """

    def __init__(self, root_dir, subset="LibriTTS/dev-clean"):
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
        try:
            waveform, sample_rate = torchaudio.load(wav_path, format="wav")
        except RuntimeError as e:
            print(f"Error loading {wav_path}: {e}")
            return None, None, None  # Return None if there's an error

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return waveform, sample_rate, text

dataset = LibriTTSDataset(root_dir="./data")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

print(f"Dataset size: {len(dataset)}")
waveform, sample_rate, text = dataset[0]
if waveform is not None:  # Check if waveform is loaded successfully
    print(f"Sample rate: {sample_rate}, Waveform shape: {waveform.shape}, Text sample: {text}")
else:
    print("Failed to load the sample.")

#Preprocessing the data

from model.text_preprocess import preprocess as text_preprocess
from model.audio_preprocess import preprocess as audio_preprocess
from tqdm import tqdm

if __name__ == "__main__":
    pbar = tqdm(total=len(dataset), desc="Processing data", unit=" samples")
    testInt = 100
    for i in range(testInt):
    # for i in range(len(dataset)):
        audio_path, text_path = dataset.metadata[i][0], dataset.metadata[i][1]
        text_preprocess(text_path)
        audio_preprocess(audio_path)
        pbar.update(1)
        if pbar.n % int(len(dataset) / 100) == 0:
            pbar.set_postfix({"Percentage": f"{pbar.n / len(dataset) * 100:.2f}%"})


    # audio_path, text_path = dataset.metadata[i][0], dataset.metadata[i][1]
    # print(f"Processing sample {i+1} of {len(dataset)}")
    # text_preprocess(text_path)
    # audio_preprocess(audio_path)
    # print(f"Processed sample {i+1} of {len(dataset)} with {text_path} and {audio_path}")