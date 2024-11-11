from main.model.text_preprocess import prerocess_from_file
import os

def normalize_text(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    if not os.path.exists("./data/custom"):
        os.mkdir("./data/custom")
    with open(f"./data/custom/{text[0]}.normalized.txt", "w", encoding="utf-8") as f:
        f.write(text)


test = input("Enter text: ")
normalize_text(test)