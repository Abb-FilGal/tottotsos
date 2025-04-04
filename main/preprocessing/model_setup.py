import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
import torch
import torchaudio
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from main.preprocessing.text_preprocess import preprocess as text_preprocess
from main.preprocessing.text_preprocess import convert_to_phonemes
from main.preprocessing.audio_preprocess import preprocess as audio_preprocess


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LibriTTSDataset(Dataset):
    def __init__(self, root_dir, subset="LibriTTS/dev-clean", preprocess=False):
        self.root_dir = os.path.join(root_dir, subset)
        self.metadata = self._load_metadata()
        if preprocess:
            self._preprocess_data()

    def _load_metadata(self):
        metadata = []
        for speaker_id in os.listdir(self.root_dir):
            speaker_dir = os.path.join(self.root_dir, speaker_id)
            if os.path.isdir(speaker_dir):
                for chapter_id in os.listdir(speaker_dir):
                    chapter_dir = os.path.join(speaker_dir, chapter_id)
                    if os.path.isdir(chapter_dir):
                        for file in os.listdir(chapter_dir):
                            if file.endswith(".wav"):
                                wav_path = os.path.join(chapter_dir, file)
                                txt_path = wav_path.replace(".wav", ".normalized.txt")
                                if os.path.exists(txt_path):
                                    metadata.append((wav_path, txt_path))
        return metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        wav_path, txt_path = self.metadata[idx]
        try:
            # Normalize path separators
            wav_path = os.path.normpath(wav_path)
            txt_path = os.path.normpath(txt_path)

            # Load preprocessed data if available
            processed_audio_path = wav_path.replace(
                os.path.normpath("./data/LibriTTS/dev-clean"),
                os.path.normpath("./data/processed")
            )
            processed_audio_path = os.path.join(
                os.path.dirname(processed_audio_path),
                "training_data",
                os.path.basename(wav_path).replace(".wav", ".npz")
            )
            
            if os.path.exists(processed_audio_path):
                data = np.load(processed_audio_path)
                melspec, audio = data['melspec'], data['audio']
            else:
                if not os.path.exists(wav_path):
                    raise FileNotFoundError(f"Audio file not found: {wav_path}")
                melspec, audio = audio_preprocess(wav_path)
                if melspec is None or audio is None:
                    raise ValueError(f"audio_preprocess returned None for {wav_path}")

            if not os.path.exists(txt_path):
                raise FileNotFoundError(f"Text file not found: {txt_path}")
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            # Load phoneme mappings
            mappings_folder = os.path.join(
                os.path.dirname(txt_path).replace(
                    os.path.normpath("./data/LibriTTS/dev-clean"),
                    os.path.normpath("./data/processed")
                ),
                "mappings"
            )
            file_name = os.path.basename(txt_path).replace(".normalized.txt", "")
            phoneme_mapping_path = os.path.join(mappings_folder, f"{file_name}_phoneme_to_id.json")
            if not os.path.exists(phoneme_mapping_path):
                raise FileNotFoundError(f"Phoneme mapping file not found: {phoneme_mapping_path}")
            with open(phoneme_mapping_path, "r") as f:
                phoneme_to_id = json.load(f)

            # Convert text to phoneme IDs
            phonemes = convert_to_phonemes(text)
            phoneme_ids = [phoneme_to_id.get(p, phoneme_to_id['<UNK>']) for p in phonemes]

            return torch.tensor(melspec), torch.tensor(phoneme_ids), text
        except Exception as e:
            print(f"Error processing sample {idx} ({wav_path}): {str(e)}")
            # Skip this sample and move to the next one
            return self.__getitem__((idx + 1) % len(self))

    def _preprocess_data(self):
        pbar = tqdm(total=len(self.metadata), desc="Preprocessing data", unit=" samples")
        for wav_path, txt_path in self.metadata:
            try:
                audio_preprocess(wav_path)
                text_preprocess(txt_path)
            except Exception as e:
                print(f"Error preprocessing {wav_path} or {txt_path}: {e}")
            pbar.update(1)
        pbar.close()

def initialize_dataset(root_dir="./data", subset="LibriTTS/dev-clean", batch_size=32, num_workers=4, preprocess=False):
    dataset = LibriTTSDataset(root_dir=root_dir, subset=subset, preprocess=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    print(f"Dataset size: {len(dataset)}")
    melspec, phoneme_ids, text = dataset[0]
    if melspec is not None:
        print(f"Mel spectrogram shape: {melspec.shape}, Phoneme IDs shape: {phoneme_ids.shape}, Text sample: {text}")
    else:
        print("Failed to load the sample.")
    
    return dataset, dataloader

if __name__ == "__main__":
    # Download the dataset if it doesn't exist
    torchaudio.datasets.LIBRITTS(root="./data", url="dev-clean", download=True)
    
    # Initialize the dataset and dataloader
    dataset, dataloader = initialize_dataset(preprocess=False)

    # Example of iterating through the dataloader
    for batch_idx, (melspecs, phoneme_ids, texts) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Mel spectrograms shape: {melspecs.shape}")
        print(f"Phoneme IDs shape: {phoneme_ids.shape}")
        print(f"Text samples: {texts[:2]}")  # Print first two text samples
        break  # Just print the first batch as an example

