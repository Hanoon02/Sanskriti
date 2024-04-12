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

predicted_values = ['image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'image8', 'image9', 'image10']
actual_values = ['image1', 'image3', 'image5', 'image7', 'image9']
evaluation = ImageEvaluation(predicted_values, actual_values)
evaluation.plot()
