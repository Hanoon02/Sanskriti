import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision.models import resnet18
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import string
import statistics
import os
import random


class HierarchicalResNet(nn.Module):
    def __init__(self, num_classes):
        super(HierarchicalResNet, self).__init__()
        self.resnet = resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fc_class = nn.Linear(num_ftrs, num_classes) 
        
    def forward(self, x):
        features = self.resnet(x)
        class_output = self.fc_class(features)
        return class_output
    
class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        self.resnet = resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)
    
class ImageInputData:
    def __init__(self):
        self.hierarchical_model = HierarchicalResNet(3)  
        self.hierarchical_model.load_state_dict(torch.load('models/HierarchicalClassificationModel.pth'))
        self.hierarchical_model.eval()
        self.hierarchical_class_names = ['Dance', 'Monuments', 'Paintings']
        self.kmeans_model_path = 'models/clusters/Warli.pkl'
        self.kmeans_model = joblib.load(self.kmeans_model_path)
        
    def get_class(self, image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),
        ])
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)  
        with torch.no_grad():
            output = self.hierarchical_model(image)
        _, predicted = torch.max(output, 1)
        hierarchical_predicted_label = self.hierarchical_class_names[predicted.item()]
        if hierarchical_predicted_label == "Paintings":
            painting_train_dataset = ImageFolder(root='../Image Data/Paintings/training', transform=transform)
            painting_class_names = painting_train_dataset.classes
            painting_model = CustomResNet(len(painting_class_names))  
            painting_model.load_state_dict(torch.load('models/Painting.pth'))
            painting_model.eval()
            painting_output = painting_model(image)
            _, painting_predicted = torch.max(painting_output, 1)
            painting_predicted_class = painting_class_names[painting_predicted.item()]
            return painting_predicted_class
        elif hierarchical_predicted_label == "Monuments":
            monuments_train_dataset = ImageFolder(root='../Image Data/Monuments/train', transform=transform)
            monuments_class_names = monuments_train_dataset.classes
            monuments_model = CustomResNet(len(monuments_class_names))  
            monuments_model.load_state_dict(torch.load('models/Monuments.pth'))
            monuments_model.eval()
            monuments_output = monuments_model(image)
            _, monuments_predicted = torch.max(monuments_output, 1)
            monuments_predicted_class = monuments_class_names[monuments_predicted.item()]
            return monuments_predicted_class
        else:
            dance_train_dataset = ImageFolder(root='../Image Data/Dance/train', transform=transform)
            dance_class_names = dance_train_dataset.classes
            dance_model = CustomResNet(len(dance_class_names))  
            dance_model.load_state_dict(torch.load('models/Dance.pth'))
            dance_model.eval()
            dance_output = dance_model(image)
            _, dance_predicted = torch.max(dance_output, 1)
            dance_predicted_class = dance_class_names[dance_predicted.item()]
            return dance_predicted_class
        
    def preprocess_image(self, image_path, target_size=(299, 299)):
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image
    
    def calculate_similarity(self, image_path1, image_path2):
        img1 = Image.open(image_path1).convert('RGB').resize((299, 299))
        img2 = Image.open(image_path2).convert('RGB').resize((299, 299))
        img1_array = np.array(img1).reshape(1, -1)
        img2_array = np.array(img2).reshape(1, -1)
        similarity = cosine_similarity(img1_array, img2_array)
        return similarity[0][0]
        
    def get_similiar_images(self, image_path):
        pretrained_model = InceptionV3(weights=None, include_top=False)
        x = pretrained_model.output
        x = GlobalAveragePooling2D()(x)
        model = Model(inputs=pretrained_model.input, outputs=x)
        image = self.preprocess_image(image_path)
        image_features = model.predict(image)
        cluster_label = self.kmeans_model.predict(image_features)
        csv_path = 'data/cluster/Warli.csv'
        df = pd.read_csv(csv_path)
        relevant_paths = df[df['Cluster_Label'] == cluster_label[0]]['Image_Path']
        similarities = {}
        for path in relevant_paths:
            similarity = self.calculate_similarity(image_path, path)
            similarities[path] = similarity
        top_3_paths = sorted(similarities, key=lambda x: similarities[x], reverse=True)[:3]
        return top_3_paths
    
class BertModelText:
    def __init__(self, input):
        self.input = input
        self.wikipedia_url = self.generate_wikipedia_url('Indian Art')
        self.content = self.fetch_website_content(self.wikipedia_url)
        
    def fetch_website_content(self, url):
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            return ' '.join([p.get_text() for p in soup.find_all('p')])
        except Exception as e:
            print(f"Error fetching website content: {e}")
        return ""

    def generate_wikipedia_url(self, search_query):
        wikipedia_api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': search_query
        }
        try:
            response = requests.get(wikipedia_api_url, params=params)
            response.raise_for_status()
            data = response.json()
            page_id = data['query']['search'][0]['pageid']
            return f"https://en.wikipedia.org/wiki?curid={page_id}"
        except Exception as e:
            print(f"Error fetching Wikipedia URL: {e}")
        return None

    def answer_question(self, question, context):
        qa_pipeline = pipeline('question-answering', model='twmkn9/bert-base-uncased-squad2')
        return qa_pipeline(question=question, context=context)['answer']
    
    def compute(self):
        return self.answer_question(self.input, self.content)

class Chatbot:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.image_keywords = ['image', 'images', 'photo', 'images', 'picture','pictures' ,'photograph', 'photos'] 

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        return tokens

    def contains_image_keywords(self, text):
        tokens = self.preprocess_text(text)
        for token in tokens:
            if any(re.search(r'\b{}\b'.format(keyword), token) for keyword in self.image_keywords):
                return True
        return False

class TextToImage:
    def __init__(self):
        self.df = pd.read_csv('data/image_text_mapping.csv')

    def get_key_terms(self, query):
        key_terms = self.process_query(query)
        return key_terms

    def process_query(self, query):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(query.lower())
        tokens = [token for token in tokens if token not in string.punctuation and token not in stop_words]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        processed_query = ' '.join(tokens)
        return processed_query

    def find_most_relevant_label(self, query, top_n=10):
        key_terms_str = self.get_key_terms(query)
        self.df['Key_Words'] = self.df['Key_Words'].str.replace("'", "").str.strip()
        self.df['Key_Words'] = self.df['Key_Words'].str[1:-1].str.split(', ')
        corpus = self.df['Key_Words']
        corpus_str = [' '.join(words) for words in corpus]
        corpus_str.append(key_terms_str)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus_str)
        similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])
        top_indices = similarities.argsort(axis=None)[-top_n:][::-1]
        top_labels = [self.df.iloc[idx]['Label'] for idx in top_indices]
        # print("APPLE", top_labels)
        most_common_label = statistics.mode(top_labels)
        most_common_class = self.df[self.df['Label'] == most_common_label]['Class'].iloc[0]
        return most_common_class, most_common_label
        
    def fetch_img(self, pred_class, pred_label):
        base_dir = ''
        if pred_class == 'Paintings':
            base_dir = '../Image Data/Paintings/training/' + pred_label
        else:
            base_dir = '../Image Data/' + pred_class + '/train/' + pred_label
        image_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
        random_image_paths = random.sample(image_files, min(3, len(image_files)))
        return random_image_paths