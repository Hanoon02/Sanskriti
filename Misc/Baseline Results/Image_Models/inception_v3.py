import os
from PIL import Image, ImageEnhance
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
import pickle
import pandas as pd

# Initialize the InceptionV3 model globally
inception_model = InceptionV3(weights='imagenet', include_top=False)
model = Model(inputs=inception_model.input, outputs=inception_model.output)

def preprocess_and_extract_features(image_path):
    """Preprocess an image from a local path and extract features using InceptionV3."""
    try:
        image = Image.open(image_path).convert('RGB')
        # InceptionV3 uses 299x299 images
        image = image.resize((299, 299)) 
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)
        image_array = np.array(image)
        image_array = preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0)
        features = model.predict(image_array)
        # Flatten the features to a 1D array
        return (image_path, features.reshape(features.shape[0], -1).tolist())
    except Exception as e:
        print(f"Error processing image from path {image_path}: {e}")
        return (image_path, None)

def save_data(data, file_name):
    """Serialize data to a file."""
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

if __name__ == "__main__":
    # Load the DataFrame
    df = pd.read_csv('/Users/dhyanpatel/Desktop/IR/image-similarity/train.csv')  # Update the path to where your CSV file is located

    # Directly use the 'path' column for image paths
    image_paths = df['path'].tolist()

    # Process images and extract features
    image_features_with_path = [preprocess_and_extract_features(path) for path in image_paths]

    # Save the results
    save_data(image_features_with_path, '/Users/dhyanpatel/Desktop/IR/image-similarity/       _image_features.pkl')  # Update the path to where you want to save your output file
