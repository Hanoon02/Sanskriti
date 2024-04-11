class ImageToImageEvaluation:
    def __init__(self, model):
        self.model = model
        self.testing_dictionary = {
            "class1": "actual_image1.jpg",
            "class2": "actual_image2.jpg",
            "class3": "actual_image3.jpg",
            "class4": "actual_image4.jpg",
            "class5": "actual_image5.jpg",
            "class6": "actual_image6.jpg"
        }
    
    def predict_images(self, input_images):
        predictions = {}
        for image_name in input_images:
            predicted_class = self.model.predict_class(image_name)
            predictions[image_name] = predicted_class
        return predictions
    
    def calculate_recall(self, num):
        top_predictions = list(self.predictions)[:num]
        verification_images = list(self.testing_dictionary.values())
        correct_predictions = sum(1 for image in top_predictions if image in verification_images)
        recall = correct_predictions / min(num, len(verification_images))
        return recall
    
    def calculate_precision(self, num):
        top_predictions = list(self.predictions)[:num]
        verification_images = list(self.testing_dictionary.values())
        relevant_predictions = sum(1 for image in top_predictions if image in verification_images)
        precision = relevant_predictions / num if num > 0 else 0
        return precision
    
    def evaluate(self):
        input_images = list(self.testing_dictionary.keys())
        self.predictions = self.predict_images(input_images)
        recall_at_5 = self.calculate_recall(5)
        recall_at_10 = self.calculate_recall(10)
        recall_at_15 = self.calculate_recall(15)
        precision_at_5 = self.calculate_precision(5)
        precision_at_10 = self.calculate_precision(10)
        precision_at_15 = self.calculate_precision(15)
        print("Recall@5:", recall_at_5)
        print("Recall@10:", recall_at_10)
        print("Recall@15:", recall_at_15)
        print("Precision@5:", precision_at_5)
        print("Precision@10:", precision_at_10)
        print("Precision@15:", precision_at_15)
