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

def create_phoneme_mapping(file_path):
    phoneme_to_id = {}
    id_to_phoneme = {}
    current_id = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            phonemes = convert_to_phonemes(text)
            for phoneme in phonemes:
                if phoneme not in phoneme_to_id:
                    phoneme_to_id[phoneme] = current_id
                    id_to_phoneme[current_id] = phoneme
                    current_id += 1

    return phoneme_to_id, id_to_phoneme

def save_mappings(phoneme_to_id, id_to_phoneme, file_path):
    mappings_folder = os.path.join("./data", os.path.dirname(file_path), "mappings")
    file_name = os.path.basename(file_path).replace(".txt", "")
    mappings_folder = os.path.join(mappings_folder, file_name)
    if not os.path.exists(mappings_folder):
        os.makedirs(mappings_folder)
    if os.path.exists(os.path.join(mappings_folder, "phoneme_to_id.json")) and os.path.exists(os.path.join(mappings_folder, "id_to_phoneme.json")):
        print(f"Skipping phoneme mappings because it already exists.")
        return
    with open(os.path.join(mappings_folder, f"{file_name}phoneme_to_id.json"), "w") as f:
        json.dump(phoneme_to_id, f)

    with open(os.path.join(mappings_folder, f"{file_name}id_to_phoneme.json"), "w") as f:
        json.dump(id_to_phoneme, f)

def preprocess(file_path):
    phoneme_to_id, id_to_phoneme = create_phoneme_mapping(file_path)
    save_mappings(phoneme_to_id, id_to_phoneme, file_path)

