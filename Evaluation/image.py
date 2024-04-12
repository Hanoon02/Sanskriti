testing_dictionary = {}
predictions = []
def calculate_recall(num):
    top_predictions = list(predictions)[:num]
    verification_images = list(testing_dictionary.values())
    correct_predictions = sum(1 for image in top_predictions if image in verification_images)
    recall = correct_predictions / min(num, len(verification_images))
    return recall

def calculate_precision(num):
    top_predictions = list(predictions)[:num]
    verification_images = list(testing_dictionary.values())
    relevant_predictions = sum(1 for image in top_predictions if image in verification_images)
    precision = relevant_predictions / num if num > 0 else 0
    return precision