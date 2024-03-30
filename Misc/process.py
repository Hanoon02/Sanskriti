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

raw_data_directory = "RawData"
if not os.path.exists(raw_data_directory):
    print("Error: RawData directory does not exist.")
    exit()

clean_data_directory = "CleanData"
if not os.path.exists(clean_data_directory):
    os.makedirs(clean_data_directory)

for filename in os.listdir(raw_data_directory):
    raw_file_path = os.path.join(raw_data_directory, filename)
    clean_file_path = os.path.join(clean_data_directory, filename)
    with open(raw_file_path, "r", encoding="utf-8") as raw_file:
        raw_text = raw_file.read()
    preprocessed_text = preprocess_text(raw_text)
    with open(clean_file_path, "w", encoding="utf-8") as clean_file:
        clean_file.write(" ".join(preprocessed_text))

print("Preprocessing complete. CleanData directory contains preprocessed files.")
