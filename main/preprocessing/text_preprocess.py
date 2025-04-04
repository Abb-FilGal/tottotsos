from g2p_en import G2p
import nltk
import json
import os

def check_nltk_resources():
    """
    Check and download required NLTK resources.
    """
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

g2p = G2p()

def convert_to_phonemes(text):
    """
    Convert text to phonemes.

    Args:
        text (str): The input text.

    Returns:
        list: A list of phonemes.
    """
    check_nltk_resources()
    return g2p(text)

def create_phoneme_mapping(file_path):
    """
    Create phoneme-to-id and id-to-phoneme mappings from a text file.

    Args:
        file_path (str): The path to the text file.

    Returns:
        tuple: A tuple containing phoneme_to_id and id_to_phoneme dictionaries.
    """
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
    """
    Save phoneme mappings to JSON files.

    Args:
        phoneme_to_id (dict): The phoneme-to-id mapping.
        id_to_phoneme (dict): The id-to-phoneme mapping.
        file_path (str): The path to the original text file.
    """
    if "LibriTTS" in file_path:
        map_name = file_path.replace("./data/LibriTTS/dev-clean", "processed")
    else:
        map_name = os.path.join("processed", os.path.basename(file_path))
        with open("log.txt", "a") as f:
            f.write(f"{map_name} was not in training data\n")

    mappings_folder = os.path.join("./data", os.path.dirname(map_name), "mappings")
    file_name = os.path.basename(file_path).replace(".normalized.txt", "")
    mappings_folder = os.path.join(mappings_folder, file_name)

    if not os.path.exists(mappings_folder):
        os.makedirs(mappings_folder)

    phoneme_to_id_path = os.path.join(mappings_folder, f"{file_name}_phoneme_to_id.json")
    id_to_phoneme_path = os.path.join(mappings_folder, f"{file_name}_id_to_phoneme.json")

    if os.path.exists(phoneme_to_id_path) and os.path.exists(id_to_phoneme_path):
        # print(f"Skipping phoneme mappings for {file_name} because they already exist.")
        return

    with open(phoneme_to_id_path, "w") as f:
        json.dump(phoneme_to_id, f)

    with open(id_to_phoneme_path, "w") as f:
        json.dump(id_to_phoneme, f)

def preprocess(file_path):
    """
    Preprocess a text file by creating and saving phoneme mappings.

    Args:
        file_path (str): The path to the text file to preprocess.
    """
    phoneme_to_id, id_to_phoneme = create_phoneme_mapping(file_path)
    save_mappings(phoneme_to_id, id_to_phoneme, file_path)

def preprocess_from_file(file):
    """
    Preprocess a single text file.

    Args:
        file (str): The path to the text file.
    """
    path = os.path.abspath(file)
    print("Processing file:", path)
    preprocess(path)

def preprocess_from_folder(folder):
    """
    Preprocess all text files in a folder.

    Args:
        folder (str): The path to the folder containing text files.
    """
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            preprocess_from_file(os.path.join(folder, file))

if __name__ == "__main__":
    # Example usage
    preprocess_from_file("./data/LibriTTS/dev-clean/84/121123/84_121123_000001_000000.normalized.txt")

