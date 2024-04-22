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
import os
import random
import os
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoModel,
    AutoTokenizer,
)
from sentence_transformers import SentenceTransformer, util
from transformers import MarianMTModel, MarianTokenizer
import torch
from sklearn.preprocessing import LabelEncoder


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
        self.hierarchical_model.load_state_dict(
            torch.load("models/HierarchicalClassificationModel.pth")
        )
        self.hierarchical_model.eval()
        self.hierarchical_class_names = ["Dance", "Monuments", "Paintings"]

    def get_class(self, image_path):
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.hierarchical_model(image)
        _, predicted = torch.max(output, 1)
        hierarchical_predicted_label = self.hierarchical_class_names[predicted.item()]
        if hierarchical_predicted_label == "Paintings":
            painting_train_dataset = ImageFolder(
                root="../Image Data/Paintings/training", transform=transform
            )
            painting_class_names = painting_train_dataset.classes
            painting_model = CustomResNet(len(painting_class_names))
            painting_model.load_state_dict(torch.load("models/Painting.pth"))
            painting_model.eval()
            painting_output = painting_model(image)
            _, painting_predicted = torch.max(painting_output, 1)
            painting_predicted_class = painting_class_names[painting_predicted.item()]
            return painting_predicted_class
        elif hierarchical_predicted_label == "Monuments":
            monuments_train_dataset = ImageFolder(
                root="../Image Data/Monuments/train", transform=transform
            )
            monuments_class_names = monuments_train_dataset.classes
            monuments_model = CustomResNet(len(monuments_class_names))
            monuments_model.load_state_dict(torch.load("models/Monuments.pth"))
            monuments_model.eval()
            monuments_output = monuments_model(image)
            _, monuments_predicted = torch.max(monuments_output, 1)
            monuments_predicted_class = monuments_class_names[
                monuments_predicted.item()
            ]
            return monuments_predicted_class
        else:
            dance_train_dataset = ImageFolder(
                root="../Image Data/Dance/train", transform=transform
            )
            dance_class_names = dance_train_dataset.classes
            dance_model = CustomResNet(len(dance_class_names))
            dance_model.load_state_dict(torch.load("models/Dance.pth"))
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
        img1 = Image.open(image_path1).convert("RGB").resize((299, 299))
        img2 = Image.open(image_path2).convert("RGB").resize((299, 299))
        img1_array = np.array(img1).reshape(1, -1)
        img2_array = np.array(img2).reshape(1, -1)
        similarity = cosine_similarity(img1_array, img2_array)
        return similarity[0][0]

    def fetch_cluster_model(self, label):
        path = "models/clusters/" + label + ".pkl"
        return joblib.load(path)

    def get_similiar_images(self, image_path, label):
        self.kmeans_model_path = "models/clusters/Warli.pkl"
        self.kmeans_model = self.fetch_cluster_model(label)
        pretrained_model = InceptionV3(weights=None, include_top=False)
        x = pretrained_model.output
        x = GlobalAveragePooling2D()(x)
        model = Model(inputs=pretrained_model.input, outputs=x)
        image = self.preprocess_image(image_path)
        image_features = model.predict(image)
        cluster_label = self.kmeans_model.predict(image_features)
        csv_path = "data/cluster/" + label + ".csv"
        df = pd.read_csv(csv_path)
        relevant_paths = df[df["Cluster_Label"] == cluster_label[0]]["Image_Path"]
        similarities = {}
        for path in relevant_paths:
            similarity = self.calculate_similarity(image_path, path)
            similarities[path] = similarity
        top_3_paths = sorted(similarities, key=lambda x: similarities[x], reverse=True)[
            :3
        ]
        return top_3_paths


class BertModelText:
    def __init__(self, input):
        self.input = input
        self.wikipedia_url = self.generate_wikipedia_url("Indian Art")
        self.content = self.fetch_website_content(self.wikipedia_url)

    def fetch_website_content(self, url):
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            return " ".join([p.get_text() for p in soup.find_all("p")])
        except Exception as e:
            print(f"Error fetching website content: {e}")
        return ""

    def generate_wikipedia_url(self, search_query):
        wikipedia_api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": search_query,
        }
        try:
            response = requests.get(wikipedia_api_url, params=params)
            response.raise_for_status()
            data = response.json()
            page_id = data["query"]["search"][0]["pageid"]
            return f"https://en.wikipedia.org/wiki?curid={page_id}"
        except Exception as e:
            print(f"Error fetching Wikipedia URL: {e}")
        return None

    def answer_question(self, question, context):
        qa_pipeline = pipeline(
            "question-answering", model="twmkn9/bert-base-uncased-squad2"
        )
        return qa_pipeline(question=question, context=context)["answer"]

    def compute(self):
        return self.answer_question(self.input, self.content)


class TextToImage:
    def __init__(self):
        self.df = pd.read_csv("data/Unique_image_text_mapping.csv")

    def get_key_terms(self, query):
        key_terms = self.process_query(query)
        return key_terms

    def process_query(self, query):
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(query.lower())
        tokens = [
            token
            for token in tokens
            if token not in string.punctuation and token not in stop_words
        ]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        processed_query = " ".join(tokens)
        return processed_query

    def find_most_relevant_label(self, query, top_n=10):
        key_terms_str = self.get_key_terms(query)
        self.df["Key_Words"] = self.df["Key_Words"].str.replace("'", "").str.strip()
        self.df["Key_Words"] = self.df["Key_Words"].str[1:-1].str.split(", ")
        corpus = self.df["Key_Words"]
        corpus_str = [" ".join(words) for words in corpus]
        corpus_str.append(key_terms_str)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus_str)
        similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        top_indices = similarities.argsort(axis=None)[-top_n:][::-1]
        top_labels = [self.df.iloc[idx]["Label"] for idx in top_indices]
        label_scores = {}
        for label, score in zip(top_labels, similarities):
            if label not in label_scores:
                label_scores[label] = 0
            label_scores[label] += score
        # print("APPLE", top_labels)
        # print("APPEL", similarities)
        most_similar_label = max(label_scores, key=label_scores.get)
        most_similar_class = self.df[self.df["Label"] == most_similar_label][
            "Class"
        ].iloc[0]
        return most_similar_class, most_similar_label

    def fetch_img(self, pred_class, pred_label):
        base_dir = ""
        if pred_class == "Paintings":
            base_dir = "../Image Data/Paintings/training/" + pred_label
        else:
            base_dir = "../Image Data/" + pred_class + "/train/" + pred_label
        image_files = [
            os.path.join(base_dir, f)
            for f in os.listdir(base_dir)
            if os.path.isfile(os.path.join(base_dir, f))
        ]
        random_image_paths = random.sample(image_files, min(3, len(image_files)))
        return random_image_paths


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class Translation:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def translate_text(self, text, src_lang, tgt_lang):
        language_codes = {
            "english": "en",
            "hindi": "hi",
            "marathi": "mr",
            "urdu": "ur",
            "bengali": "bn"
        }
        
        tgt_lang_code = language_codes.get(tgt_lang.lower(), "en") 
        
        if hasattr(self.tokenizer, 'src_lang'):  
            self.tokenizer.src_lang = src_lang
        encoded = self.tokenizer(text, return_tensors="pt")
        if hasattr(self.tokenizer, 'get_lang_id'):  
            bos_token_id = self.tokenizer.get_lang_id(tgt_lang_code)
            generated_tokens = self.model.generate(**encoded, forced_bos_token_id=bos_token_id)
        else:
            generated_tokens = self.model.generate(**encoded)
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_text

class TextInput:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-large")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def answer_question_t5(self, question, context):
        input_text = f"explain in detail: {question} context: {context}"
        inputs = self.tokenizer.encode(
            input_text, return_tensors="pt", max_length=512, truncation=True
        )
        outputs = self.model.generate(
            inputs, max_length=200, num_beams=4, early_stopping=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def models(self, input):
        context_file_path = "data/context.txt"
        with open(context_file_path, "r", encoding="utf-8") as file:
            context = file.read()

        context_sentences = context.split(". ")
        question_embedding = self.embedder.encode(input, convert_to_tensor=True)

        sentence_embeddings = self.embedder.encode(
            context_sentences, convert_to_tensor=True
        )
        cosine_scores = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0]
        most_relevant_sentence_index = torch.argmax(cosine_scores).item()
        most_relevant_sentence = context_sentences[most_relevant_sentence_index]

        try:
            answer = self.answer_question_t5(input, most_relevant_sentence)
            return answer
        except Exception as e:
            print("Error handling the input:", str(e))
class EmotionModel1(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.emotion_classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.flip_classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        emotion_output = self.emotion_classifier(output.pooler_output)
        flip_output = self.flip_classifier(output.pooler_output)
        return emotion_output, flip_output

class Feedback:
    def __init__(self):
        self.emotions = [
            "anger",
            "disgust",
            "fear",
            "joy",
            "neutral",
            "sadness",
            "surprise",
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = len(self.emotions)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.emotions)
        self.model = self.load_model(
            "../Bert_base_uncased.pth",
            EmotionModel1,
            self.num_labels,
            self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def load_model(self, model_path, model_class, num_labels, device):
        model = model_class(num_labels)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model

    def prepare_data(self, tokenizer, text, max_length=128):
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
            truncation=True,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def predict_emotions(self, text):
        data = self.prepare_data(self.tokenizer, text)
        input_ids, attention_mask = data[0], data[1]
        with torch.no_grad():
            output = self.model(input_ids, attention_mask)
        emotion_logits = output[0]
        emotion_preds = torch.argmax(emotion_logits, dim=1)
        decoded_emotion = self.label_encoder.inverse_transform([emotion_preds.item()])[
            0
        ]
        return decoded_emotion

    def get_score(self, text):
        emotion_mapping = {
            "anger" : 1,
            "disgust" : 1,
            "fear" : 1,
            "joy" : 5,
            "neutral" : 3,
            "sadness" : 2,
            "surprise" : 4,
        }
        return emotion_mapping[self.predict_emotions(text)]

from groq import Groq
translation_model = Translation("saved_models/m2m100_418M")

import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from the .env file at the start of your script
load_dotenv()

class TextInput:
    def __init__(self):
        self.context_mapping = {
            "indian_painting": "Misc/context1.txt",
            "indian_dance": "Misc/context4.txt",
            "indian_monuments_1": "Misc/context3.txt",
            "indian_monuments_2": "Misc/context2.txt"
        }
    
    def fetch_groq_response(self, user_query):
        # Determine the category of the user query
        category = self.categorize_query(user_query)
        if category is None:
            return "Unable to determine the category for the query."

        # Read the context corresponding to the category
        try:
            with open(self.context_mapping[category], "r", encoding='utf-8') as context_file:
                context = context_file.read()
        except Exception as e:
            return f"Error reading context file: {str(e)}"

        # Create Groq client using the API key for the specific category
        api_key = os.getenv(f"GROQ_API_KEY_{category}")
        if not api_key:
            return "API key not found for the specified category."

        client = Groq(api_key=api_key)
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_query},
                ],
                model="gemma-7b-it",
            )
            response = chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error fetching response from Groq: {str(e)}"

        return response

    def categorize_query(self, user_query):
        painting_keywords = ["painting", "art", "artist", "Indian painting", "canvas", "colors", "brush", "masterpiece"]
        dance_keywords = ["dance", "dancer", "dancing", "Indian dance", "bharatanatyam", "kathak", "kuchipudi", "odissi", "manipuri", "kathakali", "sattriya", "mohiniyattam"]
        monument_keywords = ["monument", "architecture", "Indian monument", "historical", "landmark", "ancient", "palace", "fort", "temple"]

        # Check if the query contains keywords related to different categories
        for keyword in painting_keywords:
            if keyword in user_query:
                return "indian_painting"
        for keyword in dance_keywords:
            if keyword in user_query:
                return "indian_dance"
        for keyword in monument_keywords:
            if keyword in user_query:
                return "indian_monuments_1"

        return None

    def process_input(self, user_query, output_type, language):
        if output_type == "Text" and language != "english":
            try:
                translate_data = translation_model.translate_text(user_query, "en", language)
                user_query = translate_data
            except Exception as e:
                print(f"Error during translation: {e}")
                return user_query
        return user_query
