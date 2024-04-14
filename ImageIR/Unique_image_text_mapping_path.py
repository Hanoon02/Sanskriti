import os
import pandas as pd
import csv
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from scipy.spatial.distance import cosine
from tqdm import tqdm

# Load InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def get_image_features(img_path):
    """Extract features from an image using InceptionV3."""
    try:
        img = image.load_img(img_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array_expanded)
        features = model.predict(img_preprocessed)
        return features.flatten()
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

DIRECTORY_PATH = '../Image Data/Image_clustering_CSV/Montuments'
output_csv_path = '../Application/data/Unique_image_text_mapping.csv'
Class='Montuments'




keyterms = {
    'Tanjore': ['temple', 'art', 'culture', 'South India', 'painting', 'Dravidian architecture', 'Chola dynasty', 'Brihadeeswarar', 'heritage'],
    'lotus_temple': ['architecture', 'delhi', 'bahai', 'worship', 'modern', 'flowerlike', 'Baháʼí House of Worship', 'open to all', 'spiritual unity'],
    'manipuri': ['dance', 'northeast', 'manipur', 'traditional', 'culture', 'ritualistic dances', 'spiritual experience', 'classical Indian dance'],
    'pattachitra': ['art', 'painting', 'odisha', 'folk', 'traditional', 'mythology', 'cloth-based scroll', 'storytelling', 'vibrant colors'],
    'alai_minar': ['architecture', 'delhi', 'unfinished', 'monument', 'history', 'Ala-ud-din Khilji', 'brick minaret', 'intended grandeur', 'Qutub complex'],
    'basilica_of_bom_jesus': ['church', 'goa', 'christianity', 'history', 'UNESCO', 'Baroque architecture', 'St. Francis Xavier', 'sacred relics', 'pilgrimage site'],
    'golden temple': ['sikh', 'amritsar', 'punjab', 'worship', 'architecture', 'Harmandir Sahib', 'holy pool', 'Guru Granth Sahib', 'pilgrimage'],
    'Warli': ['art', 'tribal', 'maharashtra', 'painting', 'traditional', 'folk style', 'rural themes', 'murals', 'cultural heritage'],
    'Kalamkari': ['art', 'textile', 'andhra pradesh', 'painting', 'traditional', 'hand-painted', 'organic dyes', 'Srikalahasti style', 'fabric storytelling'],
    'Portrait': ['art', 'painting', 'portrait', 'canvas', 'artist', 'likeness', 'individual', 'expression', 'visual representation'],
    'Khajuraho': ['temple', 'erotic', 'madhya pradesh', 'sculpture', 'architecture', 'Nagara style', 'medieval', 'Hinduism and Jainism', 'Kandariya Mahadeva'],
    'bharatanatyam': ['dance', 'classical', 'Tamil Nadu', 'performance', 'tradition', 'nritta', 'nritya', 'abhinaya', 'natya shastra'],
    'hawa mahal pics': ['architecture', 'jaipur', 'rajasthan', 'pink city', 'palace', 'wind palace', 'screen wall', 'royal women', 'red and pink sandstone'],
    'test': ['test', 'data', 'sample', 'experiment', 'analysis', 'trial', 'evaluation', 'methodology', 'results'],
    'Gateway of India': ['monument', 'mumbai', 'gateway', 'architecture', 'history', 'basalt arch', 'Mumbai Harbor', '20th-century', 'Indo-Saracenic style'],
    'Fatehpur Sikri': ['fort', 'agra', 'mughal', 'architecture', 'unesco', 'red sandstone', 'Akbar', 'abandoned city', 'buland darwaza'],
    'kathakali': ['dance', 'kerala', 'classical', 'performance', 'tradition', 'story play', 'elaborate costumes', 'facial makeup', 'gesture language'],
    'Madhubani': ['art', 'painting', 'bihar', 'folk', 'traditional', 'Mithila region', 'geometric patterns', 'epic narratives', 'natural dyes'],
    'kuchipudi': ['dance', 'andhra pradesh', 'classical', 'performance', 'tradition', 'vocal narrative', 'expressive eyes', 'quick footwork', 'drama'],
    'Sun Temple Konark': ['temple', 'odisha', 'sun god', 'sculpture', 'architecture', 'chariot shape', 'Kalinga architecture', 'world heritage', 'astronomy'],
    'iron_pillar': ['pillar', 'delhi', 'qutub minar', 'iron', 'history', 'corrosion resistance', 'metallurgical curiosity', 'Sanskrit inscriptions', 'gupta empire'],
    'alai_darwaza': ['architecture', 'delhi', 'monument', 'gateway', 'mughal', 'entrance', 'Qutub complex', 'red sandstone', 'marble decorations'],
    'Chhota_Imambara': ['mosque', 'lucknow', 'nawab', 'architecture', 'history', 'Shia Muslims', 'congregation hall', 'festivals', 'illumination'],
    'charminar': ['mosque', 'hyderabad', 'monument', 'architecture', 'landmark', 'four minarets', 'global icon', 'Indo-Islamic architecture', 'city symbol'],
    'Kangra': ['fort', 'himachal pradesh', 'kangra valley', 'history', 'architecture', 'ancient fort', 'earthquake ruins', 'strategic battles', 'maharaja sansar chand'],
    'Hawa mahal': ['architecture', 'jaipur', 'rajasthan', 'pink city', 'palace', 'high screen wall', 'rajputana design', 'honeycomb', 'breeze structure'],
    'Ellora Caves': ['cave', 'maharashtra', 'unesco', 'sculpture', 'architecture', 'rock-cut temples', 'Buddhist', 'Hindu', 'Jain', 'monolithic'],
    'tanjavur temple': ['temple', 'tamil nadu', 'architecture', 'chola', 'unesco', 'brihadisvara temple', 'great living chola temples', 'dravidian architecture', 'vimanam'],
    'Humayun_s Tomb': ['tomb', 'delhi', 'mughal', 'architecture', 'unesco', 'garden tomb', 'persian influence', 'prototype for Taj Mahal', 'indian subcontinent'],
    'mysore_palace': ['palace', 'mysore', 'karnataka', 'architecture', 'royalty', 'Wodeyar dynasty', 'dussehra', 'indo-saracenic', 'chamundi hills'],
    'India gate pics': ['war memorial', 'delhi', 'architecture', 'india', 'history', 'Amar Jawan Jyoti', 'WWI memorial', 'rajpath', 'an eternal flame','national monument', 'all india war memorial', 'patriotism'],
    'odissi': ['dance', 'odisha', 'classical', 'performance', 'tradition', 'tribhangi posture', 'mudras', 'lyrical', 'spiritual'],
    'Charar-E- Sharif': ['mosque', 'kashmir', 'sufi', 'architecture', 'history', 'wooden structure', 'muslim shrine', 'Sheikh Noor-ud-din', 'pilgrimage site'],
    'tajmahal': ['mausoleum', 'agra', 'mughal', 'architecture', 'unesco', 'ivory-white marble', 'Shah Jahan', 'Mumtaz Mahal', 'symbol of love'],
    'victoria memorial': ['memorial', 'kolkata', 'architecture', 'history', 'british', 'museum', 'queen victoria', 'british raj', 'landmark'],
    'sattriya': ['dance', 'assam', 'classical', 'performance', 'tradition', 'vaishnavism', 'monastic origins', 'ritual dance', 'bhakti movement'],
    'mohiniyattam': ['dance', 'kerala', 'classical', 'performance', 'tradition', 'lasya', 'feminine', 'graceful movements', 'vishnu'],
    'Ajanta Caves': ['cave', 'maharashtra', 'unesco', 'sculpture', 'architecture', 'buddhist art', 'rock-cut', 'murals', 'ancient'],
    'kathak': ['dance', 'uttar pradesh', 'classical', 'performance', 'tradition', 'storytelling', 'ghungroos', 'spins', 'expressions'],
    'Mural': ['art', 'painting', 'wall', 'mural', 'artist', 'public art', 'fresco', 'large scale', 'visual storytelling'],
    'qutub_minar': ['minaret', 'delhi', 'monument', 'history', 'architecture', 'victory tower', 'islamic calligraphy', 'red sandstone', 'world heritage'],
    'jamali_kamali_tomb': ['tomb', 'delhi', 'architecture', 'history', 'jamali kamali', 'mosque', 'mughal era', 'poet and sage', 'sufi saints']
}

file_exists = os.path.isfile(output_csv_path)

for filename in tqdm(os.listdir(DIRECTORY_PATH), desc="Processing files"):
    if filename.endswith('.csv'):
        csv_path = os.path.join(DIRECTORY_PATH, filename)
        data = pd.read_csv(csv_path)
        unique_clusters = data['Cluster_Label'].unique()

        for cluster in tqdm(unique_clusters, desc="Processing clusters", leave=False):
            cluster_data = data[data['Cluster_Label'] == cluster]
            if not cluster_data.empty:
                features_dict = {row['Image_Path']: get_image_features(row['Image_Path']) for _, row in tqdm(cluster_data.iterrows(), total=cluster_data.shape[0], desc="Calculating features", leave=False) if get_image_features(row['Image_Path']) is not None}
                
                if features_dict:
                    # Finding the most similar image in the cluster
                    baseline_image_path = next(iter(features_dict))
                    baseline_features = features_dict[baseline_image_path]
                    most_similar_image_path = baseline_image_path
                    lowest_cosine_distance = float('inf')
                    
                    for path, features in features_dict.items():
                        if path != baseline_image_path:
                            distance = cosine(baseline_features, features)
                            if distance < lowest_cosine_distance:
                                lowest_cosine_distance = distance
                                most_similar_image_path = path

                    label = most_similar_image_path.split('/')[-2]
                    if label in keyterms:
                        keywords = ', '.join(keyterms[label])
                        output_row = ['Dance', label, cluster, most_similar_image_path, keywords]
                        
                        with open(output_csv_path, mode='a' if file_exists else 'w', newline='', encoding='utf-8') as file:
                            writer = csv.writer(file)
                            if not file_exists:
                                writer.writerow(['Class', 'Label', 'Cluster', 'Most_Similar_Image_Path', 'Key_Words'])
                                file_exists = True
                            writer.writerow(output_row)

print("All files have been processed and results appended to:", output_csv_path)