import os
import numpy as np
import pickle
from PIL import Image, ImageEnhance
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Initialize the InceptionV3 model globally to avoid reloading
inception_model = InceptionV3(weights='imagenet', include_top=False)
model = Model(inputs=inception_model.input, outputs=inception_model.output)

def preprocess_image(image_path):
    """Preprocess an image from a local path."""
    try:
        image_path = str(image_path).strip().strip("'").strip('"')
        image = Image.open(image_path).convert('RGB')
        # Resize the image to the expected size for InceptionV3
        image = image.resize((299, 299))
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)
        return np.array(image)
    except Exception as e:
        print(f"Error processing image from path {image_path}: {e}")
        return None

def extract_features(image_array):
    """Extract features from an image array using InceptionV3."""
    if image_array is not None:
        image_array = preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0)
        features = model.predict(image_array)
        return features.reshape(features.shape[0], -1).tolist()
    return None

def cosine_similarity(vector_a, vector_b):
    """Calculate the cosine similarity between two vectors."""
    vector_a = np.array(vector_a).flatten()
    vector_b = np.array(vector_b).flatten()
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b) if (norm_a * norm_b) != 0 else 0

def find_most_similar(features_with_path, input_features, top_k):
    """Find the most similar images based on cosine similarity."""
    similarities = []
    for path, features in features_with_path:
        sim = cosine_similarity(input_features, features)
        similarities.append((path, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def display_similar_images(similar_images):
    """Display the top similar images."""
    plt.figure(figsize=(20, 10))
    for idx, (img_path, _) in enumerate(similar_images, start=1):
        plt.subplot(2, 5, idx)  # Displaying top 10 images in a 2x5 grid
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.title(f"Image {idx}")
        plt.axis('off')
    plt.show()

def main():
    # Load pre-computed data
    with open('/Users/dhyanpatel/Desktop/IR/image-similarity/inception_v3_image_features.pkl', 'rb') as file:
        image_features_with_path = pickle.load(file)

    # Prompt user for input image path
    image_path = input("Enter the image path: ")

    # Preprocess and compute features for the input image
    processed_image = preprocess_image(image_path)
    input_image_features = extract_features(processed_image) if processed_image is not None else None

    # Find the top 10 similar images based on image features
    similar_images = find_most_similar(image_features_with_path, input_image_features, top_k=10)

    # Display the images
    display_similar_images(similar_images)

if __name__ == "__main__":
    main()
