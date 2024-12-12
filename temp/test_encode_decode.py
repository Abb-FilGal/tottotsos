import sys
import os
import torch
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
os.system('cls' if os.name == 'nt' else 'clear')
from main.model.encoder import Encoder
from main.model.decoder import Decoder
from main.helper.helper import adjust_bi_to_uni_hidden_state

config = {
    "encoder": {
        "input_dim": 256,
        "conv_dim": 256,
        "lstm_dim": 128,  # Note that this is half of the decoder input size
        "num_conv_layers": 2,
        "bidirectional": True  # Ensure this is set to True for bidirectional encoder
    },
    "decoder": {
        "input_size": 256,  # Should be equal to 2 * lstm_dim if bidirectional encoder
        "lstm_dim": 128,
        "mel_dim": 256,  # This is the dimension of the mel-spectrogram output
        "teacher_forcing_ratio": 0.5,
        "max_decoder_steps": 1000
    }
}


# Initialize Encoder and Decoder with the proper config
encoder = Encoder(config["encoder"])
decoder = Decoder(config["decoder"])

# Example input for the encoder
batch_size = 16
sequence_length = 100
input_dim = 256
test_input = torch.rand(batch_size, sequence_length, input_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Forward pass through encoder
encoder_output, hidden, cell = encoder(test_input)

# Adjust BiLSTM hidden state if the encoder is bidirectional
hidden, cell = adjust_bi_to_uni_hidden_state(hidden, cell, bidirectional=True)

# Generate variable-length decoder output
decoder_input = torch.zeros(batch_size, 1, config["decoder"]["mel_dim"]).to(test_input.device)  # Start with zero vector
decoder_output = decoder(encoder_output, (hidden, cell))

print("Decoder output shape:", decoder_output.shape)
