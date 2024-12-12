import os
import json
import torch

def load_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '../../config/config.json')
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def adjust_bi_to_uni_hidden_state(hidden, cell, bidirectional=False):
    """
    Adjust hidden and cell state for bidirectional encoder to unidirectional decoder.
    """
    if bidirectional:
        # Assuming hidden and cell have shape (num_layers * num_directions, batch, hidden_size)
        num_layers = hidden.size(0) // 2
        hidden = hidden.view(num_layers, 2, -1, hidden.size(2)).sum(dim=1)
        cell = cell.view(num_layers, 2, -1, cell.size(2)).sum(dim=1)
    return hidden, cell

