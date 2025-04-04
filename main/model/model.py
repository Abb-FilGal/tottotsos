import torch
import torch.nn as nn
from main.model.encoder import Encoder
from main.model.decoder import Decoder
from main.helper.helper import adjust_bi_to_uni_hidden_state

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.encoder = Encoder(config["encoder"])
        self.decoder = Decoder(config["decoder"])
        self.bidirectional = config["encoder"]["bidirectional"]

    def forward(self, x, target_mel=None):
        # Forward pass through encoder
        encoder_output, hidden, cell = self.encoder(x)

        # Adjust BiLSTM hidden state if the encoder is bidirectional
        hidden, cell = adjust_bi_to_uni_hidden_state(hidden, cell, bidirectional=self.bidirectional)

        # Forward pass through decoder
        decoder_output = self.decoder(encoder_output, (hidden, cell), target_mel)

        return decoder_output

