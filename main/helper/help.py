import os
import json

def load_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '../../config/config.json')
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config
