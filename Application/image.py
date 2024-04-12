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
class ImageData:
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
        train_dataset = ImageFolder(root='../Image Data/Paintings/training', transform=transform)
        class_names = train_dataset.classes
        painting_model = CustomResNet(len(class_names))  
        painting_model.load_state_dict(torch.load('models/Painting.pth'))
        painting_model.eval()
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)  
        with torch.no_grad():
            output = self.hierarchical_model(image)
        _, predicted = torch.max(output, 1)
        hierarchical_predicted_label = self.hierarchical_class_names[predicted.item()]
        if hierarchical_predicted_label == "Paintings":
            painting_output = painting_model(image)
            _, painting_predicted = torch.max(painting_output, 1)
            painting_predicted_class = class_names[painting_predicted.item()]
            return painting_predicted_class
        else:
            return hierarchical_predicted_label
        
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