import csv
import os
import ast
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

def extract_key_terms(query):
    return preprocess_text(query)

def parse_key_terms(key_terms_str):
    return [term.strip() for term in key_terms_str]

def load_image_data(csv_file_name):
    data = []
    with open(csv_file_name, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            key_terms = parse_key_terms(row['Keyterms'])
            row['Keyterms'] = preprocess_text(','.join(key_terms))
            data.append(row)
    return data

def find_most_relevant_images(user_query, image_data):
    key_terms = extract_key_terms(user_query)
    
    corpus = [item['Keyterms'] for item in image_data]
    corpus.append(key_terms)
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    
    similarity_matrix = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1])
    most_similar_index = similarity_matrix.argmax()
    
    return image_data[most_similar_index]['Class'], image_data[most_similar_index]['Path']

csv_file_name = 'Image_Keyterms.csv'
image_data = load_image_data(csv_file_name)

user_query = input("Enter your query: ")
image_class, most_relevant_image_path = find_most_relevant_images(user_query, image_data)
print("Most relevant image: {} - {}".format(image_class, most_relevant_image_path))
