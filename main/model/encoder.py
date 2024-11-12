import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from main.helper.help import load_config

class CustomEncoder(nn.Module):
    def __init__(self, input_dim, conv_dim, lstm_dim, num_conv_layers=2):
        super(CustomEncoder, self).__init__()
        
        # Convolutional Layers for local feature extraction
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_dim if i == 0 else conv_dim, conv_dim, kernel_size=3, padding=1)
            for i in range(num_conv_layers)
        ])
        
        # Bidirectional LSTM for capturing long-range dependencies
        self.bilstm = nn.LSTM(conv_dim, lstm_dim, bidirectional=True, batch_first=True)
        
        # Simple Attention Mechanism using a linear layer
        self.attn = nn.Linear(lstm_dim * 2, 1)  # BiLSTM doubles the hidden size
        
    def forward(self, x):
        # (batch_size, sequence_length, input_dim) -> (batch_size, input_dim, sequence_length)
        x = x.permute(0, 2, 1)
        
        # Apply each convolutional layer
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        
        # (batch_size, conv_dim, sequence_length) -> (batch_size, sequence_length, conv_dim)
        x = x.permute(0, 2, 1)
        
        # Bidirectional LSTM
        lstm_out, (hidden, cell) = self.bilstm(x)
        
        # Simple Attention Mechanism
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        
        return context_vector, hidden, cell


config = load_config()
config = config["encoder"]
encoder = CustomEncoder(
    input_dim=config["input_dim"],
    conv_dim=config["conv_dim"],
    lstm_dim=config["lstm_dim"],
    # num_conv_layers=config["num_conv_layers"]
)

# Example input tensor for testing
batch_size = 32
sequence_length = 100
input_dim = config["input_dim"]
test_input = torch.rand(batch_size, sequence_length, input_dim)

# Forward pass
output, hidden, cell = encoder(test_input)
print("Encoder output shape:", output.shape)
