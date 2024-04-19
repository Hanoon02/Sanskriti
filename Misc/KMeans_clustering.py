# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# from skimage.io import imread
# from skimage.transform import resize
# from skimage.color import rgb2gray
# import os
# import csv

# # Parameters
# image_folder = '/Users/dhyanpatel/Desktop/IR/sanskriti/Image Data/Paintings/Madhubani'
# image_shape = (64, 64)
# n_clusters = 4
# output_csv = '/Users/dhyanpatel/Desktop/IR/sanskriti/Image Data/Paintings/Madhubani_cluster.csv'

# # Function to preprocess an image
# def preprocess_image(image_path, image_shape):
#     # Read image
#     image = imread(image_path)
#     # Convert to grayscale
#     image = rgb2gray(image)
#     # Resize image
#     image = resize(image, image_shape, anti_aliasing=True)
#     # Flatten image
#     return image.flatten()

# # Get all image paths
# image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# # Preprocess images
# image_vectors = np.array([preprocess_image(path, image_shape) for path in image_paths])

# # Normalize the data
# scaler = StandardScaler()
# image_vectors_scaled = scaler.fit_transform(image_vectors)

# # Dimensionality Reduction with PCA
# pca = PCA(n_components=10)  # retain 90% of variance
# image_vectors_pca = pca.fit_transform(image_vectors_scaled)

# # Apply k-means clustering
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# kmeans.fit(image_vectors_pca)

# # Get cluster assignments for each image
# cluster_assignments = kmeans.labels_

# # Write to CSV
# with open(output_csv, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Image Path', 'Cluster'])
#     for path, cluster in zip(image_paths, cluster_assignments):
#         writer.writerow([path, cluster])

# print(f"Cluster assignments have been saved to {output_csv}.")



##########################################################################################

# KMeans_clustering Model

# import numpy as np
# import os
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.keras.models import Model
# from joblib import dump
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from tqdm import tqdm

# # Parameters
# image_folder = '/Users/dhyanpatel/Desktop/IR/sanskriti/Image Data/Paintings/Madhubani'
# n_clusters = 4
# model_path = '/Users/dhyanpatel/Desktop/IR/sanskriti/Image Data/Paintings/Madhubani/kmeans_model.pkl'
# scaler_path = '/Users/dhyanpatel/Desktop/IR/sanskriti/Image Data/Paintings/Madhubani/scaler.pkl'

# # Load VGG16 model, excluding the top (classification) layer
# base_model = VGG16(weights='imagenet', include_top=False)
# model = Model(inputs=base_model.input, outputs=base_model.output)

# # Function to preprocess an image and extract features using VGG16
# def extract_features(image_path, model):
#     image = load_img(image_path, target_size=(224, 224))
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=0)
#     image = preprocess_input(image)
#     features = model.predict(image)
#     return features.flatten()

# # Get all image paths
# image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
#                if os.path.isfile(os.path.join(image_folder, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'))]

# # Extract features for all images, with progress bar
# features = np.array([extract_features(path, model) for path in tqdm(image_paths, desc="Extracting features")])

# # Standardize features
# scaler = StandardScaler()
# features = scaler.fit_transform(features)

# # Train KMeans model
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# kmeans.fit(features)

# # Save the KMeans model and the scaler
# dump(kmeans, model_path)
# dump(scaler, scaler_path)

# print(f"KMeans model and scaler saved.")


##########################################################################################

# KMeans_clustering classifying the image

from joblib import load
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np

# Load the trained KMeans model and scaler
model_path = '/Users/dhyanpatel/Desktop/IR/sanskriti/Image Data/Paintings/Madhubani/kmeans_model.pkl'
scaler_path = '/Users/dhyanpatel/Desktop/IR/sanskriti/Image Data/Paintings/Madhubani/scaler.pkl'
kmeans = load(model_path)
scaler = load(scaler_path)

# Load the VGG16 model for feature extraction
base_model = VGG16(weights='imagenet', include_top=False)
feature_model = Model(inputs=base_model.input, outputs=base_model.output)

# Function to preprocess an image and extract features using VGG16
def extract_features(image_path, model):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features.flatten()

# Function to predict the cluster for a new image using extracted features
def predict_cluster(image_path, feature_model, kmeans_model, scaler):
    features = extract_features(image_path, feature_model)
    # Ensure features are reshaped to a 2D array. The reshape to (1, -1) should already do this,
    # but it's critical to confirm there's no additional unintended dimension.
    features = features.reshape(1, -1)  # This ensures a 2D array shape
    features_scaled = scaler.transform(features)  # Scale features
    cluster = kmeans_model.predict(features_scaled)  # Predict cluster
    return cluster[0]


# Example usage
image_path = input("Enter the path to the image: ")
cluster = predict_cluster(image_path, feature_model, kmeans, scaler)
print(f"The image belongs to cluster {cluster}")
