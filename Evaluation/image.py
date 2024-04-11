input_images = ["predicted_image1.jpg", "predicted_image2.jpg", "predicted_image3.jpg", "predicted_image4.jpg", "predicted_image5.jpg", "predicted_image6.jpg"]
verification_images = ["predicted_image1.jpg", "actual_image2.jpg", "predicted_image3.jpg", "actual_image4.jpg", "predicted_image5.jpg"]
    
def calculate_recall(num, input_images, verification_images):
    top_predictions = input_images[:num] 
    correct_predictions = sum(1 for image in top_predictions if image in verification_images)
    recall = correct_predictions / min(num, len(verification_images)) 
    return recall

def calculate_precision(num, input_images, verification_images):
    top_predictions = input_images[:num] 
    relevant_predictions = sum(1 for image in top_predictions if image in verification_images)
    precision = relevant_predictions / num if num > 0 else 0
    return precision

recall_at_5 = calculate_recall(5, input_images, verification_images)
recall_at_10 = calculate_recall(10, input_images, verification_images)
recall_at_15 = calculate_recall(15, input_images, verification_images)
print("Recall@5:", recall_at_5)
print("Recall@10:", recall_at_10)
print("Recall@15:", recall_at_15)

precision_at_5 = calculate_precision(5, input_images, verification_images)
precision_at_10 = calculate_precision(10, input_images, verification_images)
precision_at_15 = calculate_precision(15, input_images, verification_images)

print("Precision@5:", precision_at_5)
print("Precision@10:", precision_at_10)
print("Precision@15:", precision_at_15)