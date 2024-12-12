import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.input_dim = config["input_dim"]
        self.conv_dim = config["conv_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.num_conv_layers = config["num_conv_layers"]
        
        # Convolutional Layers for local feature extraction
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(self.input_dim if i == 0 else self.conv_dim, self.conv_dim, kernel_size=3, padding=1)
            for i in range(self.num_conv_layers)
        ])
        
        # Bidirectional LSTM for capturing long-range dependencies
        self.bilstm = nn.LSTM(self.conv_dim, self.lstm_dim, bidirectional=True, batch_first=True)
        
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
        
        return lstm_out, hidden, cell
