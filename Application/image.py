import pickle
import numpy as np
from PIL import Image
from transformers import BeitForImageClassification, BeitFeatureExtractor
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import requests
from io import BytesIO

class MonumentImage:
    def __init__(self, image_path):
        self.image_path = image_path
        self.model_path = 'models/fine_tuned_model'
        self.class_names = ["Ajanta Caves", "Charar-E- Sharif", "Chhota Imambara", "Ellora Caves", "Fatehpur Sikiri",
                            "Hawa Mahal", "Gateway of India", "Khajuraho", "Sun Temple Konark", "Alai Darwaza",
                            "Alai Minar", "Basilica of Bom Jesus", "Charminar", "Golden Temple", "Iron Pillar",
                            "Jamali Kamali Tomb", "Lotus Temple", "Mysore Palace", "Qutub Minar", "Taj Mahal",
                            "Tanjavur Temple", "Victoria Memorial"]
        
    def predict_image_class(self, image_input):
        model_path = self.model_path
        model = BeitForImageClassification.from_pretrained(model_path)
        feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224')
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        if image_input.startswith('http'):
            response = requests.get(image_input)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_input).convert("RGB")
        transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0) 
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image)
            preds = outputs.logits.softmax(dim=-1)
            predicted_index = preds.argmax(1).item()
        if predicted_index < len(self.class_names):
            predicted_class = self.class_names[predicted_index]
        else:
            predicted_class = "Unknown Class"
        return f"Predicted class for the image: {predicted_class}"
        
    def compute(self):
        try:
            class_pred = self.predict_image_class(self.image_path)
        except Exception as e:
            return f"Error: {str(e)}"
        return class_pred
