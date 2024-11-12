from main.preprocessing.text_preprocess import preprocess_from_file, preprocess_from_folder
import os

def normalize_text(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    file_name = f"{text.split()[0]}_{text.split()[1]}"
    if not os.path.exists("./data/custom"):
        os.mkdir("./data/custom")
    with open(f"./data/custom/{file_name}.normalized.txt", "w", encoding="utf-8") as f:
        f.write(text)
    return file_name


# test = input("Enter text: ")
# normalize_text(test)

# prerocess_from_file(f"./data/custom/{test[0]}.normalized.txt")
# os.remove(f"./data/custom/{test[0]}.normalized.txt")

def custom_input():
    try:
        option = input("Do you want to TTS a file OR a folder? (0 for file, 1 for folder, 2 for terminal input): ")
        if option.lower().strip() == "0":
            file_path = input("Enter the path to the file: ")
            preprocess_from_file(file_path)
        elif option.lower().strip() == "1":
            folder_path = input("Enter the path to the folder: ")
            preprocess_from_folder(folder_path)
        elif option.lower().strip() == "2":
            text = input("Enter the text: ")
            file_name = normalize_text(text)
            preprocess_from_file(f"./data/custom/{file_name}.normalized.txt")
        else:
            print("Invalid option.")
            custom_input()
    except Exception as e:
        print(f"Error: {e}")


