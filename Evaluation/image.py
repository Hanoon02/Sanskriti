import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision.models import resnet18
import numpy as np

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
    

class ImageEvaluation:
    def __init__(self, predicted_values, actual_values):
        self.predicted_values = predicted_values
        self.actual_values = actual_values
    
    def calculate_recall(self, num):
        relevant_images = self.actual_values[:num]
        relevant_predicted = [image in relevant_images for image in self.predicted_values[:num]]
        recall = sum(relevant_predicted) / len(relevant_images)
        return recall

    def calculate_precision(self, num):
        relevant_images = self.actual_values[:num]
        relevant_predicted = [image in relevant_images for image in self.predicted_values[:num]]
        precision = sum(relevant_predicted) / num
        return precision
    
    def create_matrix(self):
        recall_values = []
        precision_values = []
        for num in [5, 10, 15]:
            recall_values.append(self.calculate_recall(num))
            precision_values.append(self.calculate_precision(num))
        return np.array(recall_values), np.array(precision_values)
    
    def plot(self, title="Classification Results"):
        recall_values, precision_values = self.create_matrix()
        fig, axs = plt.subplots(2, figsize=(10, 8))
        axs[0].bar(['5', '10', '15'], recall_values, color='blue', alpha=0.7, label='Average Recall')
        axs[0].set_title(f'Average Recall for Different Subset Sizes: {title}')
        axs[0].set_xlabel('Subset Size')
        axs[0].set_ylabel('Value')
        axs[0].legend()
        axs[1].bar(['5', '10', '15'], precision_values, color='red', alpha=0.7, label='Average Precision')
        axs[1].set_title(f'Average Precision for Different Subset Sizes: {title}')
        axs[1].set_xlabel('Subset Size')
        axs[1].set_ylabel('Value')
        axs[1].legend()
        plt.tight_layout()
        plt.show()


'''
class ImageInputData:
    def __init__(self):
        self.hierarchical_model = HierarchicalResNet(3)  
        self.hierarchical_model.load_state_dict(torch.load('models/HierarchicalClassificationModel.pth'))
        self.hierarchical_model.eval()
        self.hierarchical_class_names = ['Dance', 'Monuments', 'Paintings']
        
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

            '''



# Code to get random 20 image paths from the dance folder of different classes


def select_random_images(csv_path, classes, num_samples=20):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Dictionary to store results
    selected_images = {}

    # Iterate over each class
    for cls in classes:
        # Filter the DataFrame for the current class
        class_df = df[df['Class'] == cls]
        
        # Sample 20 images randomly, or less if there aren't enough
        if len(class_df) > num_samples:
            sampled_df = class_df.sample(n=num_samples)
        else:
            sampled_df = class_df
        
        # Store the sampled data in the dictionary
        selected_images[cls] = sampled_df

    return selected_images

# Define the path to your CSV file and the classes you're interested in
csv_path = 'data/Unique_image_text_mapping.csv'
# classes = ['Monuments', 'Dance', 'Paintings']
classes = ['Dance']

# Get the selected images
random_images_by_class = select_random_images(csv_path, classes)




# Assuming HierarchicalResNet and CustomResNet are defined and imported correctly
# class HierarchicalResNet, CustomResNet need to be defined or imported above this script

class ImageInputData:
    def __init__(self):
        self.hierarchical_model = HierarchicalResNet(3)  
        self.hierarchical_model.load_state_dict(torch.load('models/HierarchicalClassificationModel.pth'))
        self.hierarchical_model.eval()
        self.hierarchical_class_names = ['Dance', 'Monuments', 'Paintings']

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
        
        # Load the model corresponding to the predicted class
        if hierarchical_predicted_label == "Paintings":
            class_model_path = 'models/Painting.pth'
            dataset_path = '../Image Data/Paintings/training'
        elif hierarchical_predicted_label == "Monuments":
            class_model_path = 'models/Monuments.pth'
            dataset_path = '../Image Data/Monuments/train'
        else:
            class_model_path = 'models/Dance.pth'
            dataset_path = '../Image Data/Dance/train'
        
        # Process using the specific model
        specific_train_dataset = ImageFolder(root=dataset_path, transform=transform)
        class_names = specific_train_dataset.classes
        model = CustomResNet(len(class_names))
        model.load_state_dict(torch.load(class_model_path))
        model.eval()
        specific_output = model(image)
        _, specific_predicted = torch.max(specific_output, 1)
        predicted_class = class_names[specific_predicted.item()]
        return hierarchical_predicted_label, predicted_class, dataset_path

# Initialize the ImageInputData class
image_classifier = ImageInputData()

# Get the predicted class based on the list of image paths
predicted_values = []

# Add the path in a list
image_paths = []
for cls in classes:
    image_paths.extend(random_images_by_class[cls]['Most_Similar_Image_Path'].tolist())

for path in image_paths:
    hierarchical_predicted_label, predicted_class, dataset_path = image_classifier.get_class(path)
    predicted_values.append(f'{predicted_class}')



# find the actual class from datapath:
actual_values = []
for path in image_paths:
    actual_values.append(path.split('/')[-2])


for actual, predicted in zip(actual_values, predicted_values):
    print(f'Actual: {actual}, Predicted: {predicted}')
    
evaluation = ImageEvaluation(predicted_values, actual_values)
evaluation.plot(title="Dance")
