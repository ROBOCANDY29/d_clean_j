import os
import ast
from multiprocessing import cpu_count

import pandas as pd
from datasets import load_dataset

from indusnlp import TextCleaner, HindiTextCleaner
import sys
import os
# Add the parent directory to the path to import the local clean module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from clean.clean import clean_text  # Import from the local clean module

# Configuration for the text cleaning
config = [
    ("handle_whitespace", None),
    ("remove_redundant_lines", None),
    ("remove_blank_lines", None),
]

# Initializing cleaners
textcleaner = TextCleaner(config)
hicleaner = HindiTextCleaner(transliterate=True)

def clean_content(example):
    conversation_str = example.get("conversations")

    if conversation_str is None:
        return {"conversations": None}

    try:
        # Safely evaluate the string to a Python object
        conversation_list = ast.literal_eval(conversation_str)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing conversation: {conversation_str}")
        print(f"Exception: {e}")
        return {"conversations": None}

    cleaned_conversations = []

    for pair in conversation_list:
        cleaned_pair = []
        for text in pair:
            if not isinstance(text, str):
                return {"conversations": None}

            # Check for bad words
            text_bad_word = clean_text(text)
            if text_bad_word is not None:
                # Clean text
                cleaned_text = hicleaner(textcleaner(text))
                cleaned_pair.append(cleaned_text)
            else:
                return {"conversations": None}  # Discard if bad words are found

        cleaned_conversations.append(cleaned_pair)

    return {"conversations": cleaned_conversations}

def process_dataset(folder_path):
    translated_folder = os.path.join(folder_path, "translated")
    cleaned_folder = os.path.join(folder_path, "cleaned")

    os.makedirs(cleaned_folder, exist_ok=True)

    for filename in os.listdir(translated_folder):
        if filename.endswith(".csv"):
            translated_file = os.path.join(translated_folder, filename)

            # Load CSV
            dataset = load_dataset('csv', data_files=translated_file)

            # Apply cleaning
            dataset = dataset.map(clean_content, num_proc=1)  # Disable multiprocessing for Windows compatibility

            # Filter out invalid or empty rows
            dataset = dataset.filter(
                lambda example: (
                    example["conversations"] is not None and
                    len(example["conversations"]) > 0 and
                    all(
                        len(sublist) > 0 and
                        all(text is not None and text.strip() != "" for text in sublist)
                        for sublist in example["conversations"]
                    )
                )
            )

            # Convert back to string for saving
            dataset = dataset.map(lambda example: {"conversations": repr(example["conversations"])})

            # Save cleaned CSV
            cleaned_file = os.path.join(cleaned_folder, filename.replace(".csv", "_cleaned.csv"))
            dataset['train'].to_csv(cleaned_file)

# Example usage
if __name__ == '__main__':
    folder_name = "D:\INDUS\data_cleaning\IndusNLPToolkit\IndusNLPToolkit\cleaned"
    process_dataset(folder_name)
