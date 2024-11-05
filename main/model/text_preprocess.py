from g2p_en import G2p
import nltk
import json
import os

def check_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger_eng')

g2p = G2p()

def convert_to_phonemes(text):
    check_nltk_resources()
    return g2p(text)

def create_phoneme_mapping(texts):
    phoneme_to_id = {}
    id_to_phoneme = {}
    current_id = 0

    for text in texts:
        phonemes = convert_to_phonemes(text)
        for phoneme in phonemes:
            if phoneme not in phoneme_to_id:
                phoneme_to_id[phoneme] = current_id
                id_to_phoneme[current_id] = phoneme
                current_id += 1

    return phoneme_to_id, id_to_phoneme

# Example usage
texts = [
    "Hello, how are you?",
    "This is a test sentence.",
    "Text to speech conversion."
]

phoneme_to_id, id_to_phoneme = create_phoneme_mapping(texts)

def save_mappings(phoneme_to_id, id_to_phoneme):
    mappings_folder = "./data/mappings"
    if not os.path.exists(mappings_folder):
        os.makedirs(mappings_folder)
    with open("./data/mappings/phoneme_to_id.json", "w") as f:
        json.dump(phoneme_to_id, f)

    with open("./data/mappings/id_to_phoneme.json", "w") as f:
        json.dump(id_to_phoneme, f)

save_mappings(phoneme_to_id, id_to_phoneme)
