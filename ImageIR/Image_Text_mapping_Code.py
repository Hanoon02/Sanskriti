import os
import pandas as pd
import csv

DIRECTORY_PATH = '../Image Data/Image_clustering_CSV/Dance'  
output_csv_path = '../Application/data/image_text_mapping.csv'  



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

for filename in os.listdir(DIRECTORY_PATH):
    if filename.endswith('.csv'):
        csv_path = os.path.join(DIRECTORY_PATH, filename)
        data = pd.read_csv(csv_path)
        unique_clusters = data['Cluster_Label'].unique()

        output_rows = []
        Class = 'Dance'  

        for cluster in unique_clusters:
            cluster_data = data[data['Cluster_Label'] == cluster]
            if not cluster_data.empty:
                label = cluster_data.iloc[0]['Image_Path'].split('/')[-2]
                if label in keyterms:
                    keywords = str(keyterms[label]) 
                    output_rows.append([Class, label, cluster, keywords])

        with open(output_csv_path, mode='a' if file_exists else 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Class', 'Label', 'Cluster', 'Key_Words'])
                file_exists = True  
            for row in output_rows:
                writer.writerow(row)

print("All files have been processed and results appended to:", output_csv_path)