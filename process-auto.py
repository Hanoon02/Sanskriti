import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words]
    punctuation_chars = set(string.punctuation)
    tokens = [word for word in tokens if not any(char in punctuation_chars for char in word)]
    return tokens

auto_raw_data_directory = "AutoRawData"
if not os.path.exists(auto_raw_data_directory):
    print("Error: AutoRawData directory does not exist.")
    exit()

auto_clean_data_directory = "AutoCleanData"
if not os.path.exists(auto_clean_data_directory):
    os.makedirs(auto_clean_data_directory)

for directory in os.listdir(auto_raw_data_directory):
    raw_directory_path = os.path.join(auto_raw_data_directory, directory)
    clean_directory_path = os.path.join(auto_clean_data_directory, directory)
    if not os.path.exists(clean_directory_path):
        os.makedirs(clean_directory_path)
    for filename in os.listdir(raw_directory_path):
        raw_file_path = os.path.join(raw_directory_path, filename)
        clean_file_path = os.path.join(clean_directory_path, filename)
        with open(raw_file_path, "r", encoding="utf-8") as raw_file:
            raw_text = raw_file.read()
        preprocessed_text = preprocess_text(raw_text)
        with open(clean_file_path, "w", encoding="utf-8") as clean_file:
            clean_file.write(" ".join(preprocessed_text))

print("Preprocessing complete. AutoCleanData directory contains preprocessed files.")
