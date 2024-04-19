import numpy as np
import matplotlib.pyplot as plt

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
        for _ in range(10):
            recall = [self.calculate_recall(num) for num in [5, 10, 15]]
            precision = [self.calculate_precision(num) for num in [5, 10, 15]]
            recall_values.append(recall)
            precision_values.append(precision)
        return np.array(recall_values), np.array(precision_values)
    
    def plot(self):
        recall_values, precision_values = self.create_matrix()
        avg_recall = np.mean(recall_values, axis=0)
        avg_precision = np.mean(precision_values, axis=0)
        fig, axs = plt.subplots(2, figsize=(10, 8))
        axs[0].bar(['5', '10', '15'], avg_recall, color='blue', alpha=0.7, label='Average Recall')
        axs[0].set_title('Average Recall for Different Subset Sizes')
        axs[0].set_xlabel('Subset Size')
        axs[0].set_ylabel('Value')
        axs[0].legend()
        axs[1].bar(['5', '10', '15'], avg_precision, color='red', alpha=0.7, label='Average Precision')
        axs[1].set_title('Average Precision for Different Subset Sizes')
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

# predicted_values = ['image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'image8', 'image9', 'image10']
# actual_values = ['image1', 'image3', 'image5', 'image7', 'image9']


# Code to get random 20 image paths from the dance folder of different classes




evaluation = ImageEvaluation()
evaluation.plot('Painting')
