from main.model_setup import LibriTTSDataset
from torch.utils.data import DataLoader


dataset = LibriTTSDataset(root_dir="./data")
batch_size = len(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

print(f"Dataset size: {len(dataset)}")

for i in range(batch_size):
    waveform, sample_rate, text = dataset[i]
    if waveform is not None:  # Check if waveform is loaded successfully
        string = f"Sample rate: {sample_rate}, Waveform shape: {waveform.shape}, Text sample: {text}"
        print(string + "\n")
        with open("output.txt", "a") as f:  # Use "a" to append to the file
            f.write(string + "\n")
    else:
        print("Failed to load the sample.")
