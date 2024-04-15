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

DIRECTORY_PATH = '../Image Data/Image_clustering_CSV/Paintings'
output_csv_path = '../Application/data/Unique_image_text_mapping.csv'
Class='Paintings'




# keyterms = {
#     'Tanjore': ['temple', 'art', 'culture', 'South India', 'painting', 'Dravidian architecture', 'Chola dynasty', 'Brihadeeswarar', 'heritage'],
#     'lotus_temple': ['architecture', 'delhi', 'bahai', 'worship', 'modern', 'flowerlike', 'Baháʼí House of Worship', 'open to all', 'spiritual unity'],
#     'pattachitra': ['art', 'painting', 'odisha', 'folk', 'traditional', 'mythology', 'cloth-based scroll', 'storytelling', 'vibrant colors'],
#     'alai_minar': ['architecture', 'delhi', 'unfinished', 'monument', 'history', 'Ala-ud-din Khilji', 'brick minaret', 'intended grandeur', 'Qutub complex'],
#     'basilica_of_bom_jesus': ['church', 'goa', 'christianity', 'history', 'UNESCO', 'Baroque architecture', 'St. Francis Xavier', 'sacred relics', 'pilgrimage site'],
#     'golden temple': ['sikh', 'amritsar', 'punjab', 'worship', 'architecture', 'Harmandir Sahib', 'holy pool', 'Guru Granth Sahib', 'pilgrimage'],
#     'Warli': ['art', 'tribal', 'maharashtra', 'painting', 'traditional', 'folk style', 'rural themes', 'murals', 'cultural heritage'],
#     'Kalamkari': ['art', 'textile', 'andhra pradesh', 'painting', 'traditional', 'hand-painted', 'organic dyes', 'Srikalahasti style', 'fabric storytelling'],
#     'Portrait': ['art', 'painting', 'portrait', 'canvas', 'artist', 'likeness', 'individual', 'expression', 'visual representation'],
#     'Khajuraho': ['temple', 'erotic', 'madhya pradesh', 'sculpture', 'architecture', 'Nagara style', 'medieval', 'Hinduism and Jainism', 'Kandariya Mahadeva'],
#     'hawa mahal pics': ['architecture', 'jaipur', 'rajasthan', 'pink city', 'palace', 'wind palace', 'screen wall', 'royal women', 'red and pink sandstone'],
#     'Gateway of India': ['monument', 'mumbai', 'gateway', 'architecture', 'history', 'basalt arch', 'Mumbai Harbor', '20th-century', 'Indo-Saracenic style'],
#     'Fatehpur Sikri': ['fort', 'agra', 'mughal', 'architecture', 'unesco', 'red sandstone', 'Akbar', 'abandoned city', 'buland darwaza'],
#     'kathak': ['dance', 'indian classical', 'kathak', 'ghungroos', 'storytelling', 'hand gestures', 'footwork', 'costumes', 'traditional attire', 'cultural heritage', 'performance'],
#     'kuchipudi': ['dance', 'south indian', 'kuchipudi', 'expressive eyes', 'storytelling', 'fluid movements', 'rhythmic footwork', 'silk sarees', 'temple dance', 'vibrant presence', 'drama'],
#     'bharatanatyam': ['dance', 'tamil nadu', 'bharatanatyam', 'mudras', 'spiritual expression', 'temple tradition', 'carnatic music', 'precise movements', 'facial expressions', 'foot tapping'],
#     'kathakali': ['dance', 'kerala', 'kathakali', 'face paint', 'headdresses', 'mythology', 'facial expressions', 'vigorous dance', 'ritual performance', 'masculine dance', 'epic narratives'],
#     'odissi': ['dance', 'east indian', 'odissi', 'tribhangi posture', 'sculptural poses', 'story narration', 'spiritual themes', 'feminine', 'group performance', 'devotional'],
#     'mohiniyattam': ['dance', 'kerala', 'mohiniyattam', 'lyrical movements', 'sensuous', 'solo performance', 'elegant costumes', 'subtle expressions', 'emotional interpretation'],
#     'sattriya': ['dance', 'assam', 'sattriya', 'devotional dance', 'vaishnavism', 'monastery tradition', 'narrative dance', 'classical form', 'ritualistic', 'ensemble performance'],
#     'manipuri': ['dance', 'northeast indian', 'manipuri', 'rounded movements', 'drum music', 'graceful', 'ethereal costumes', 'tandava and lasya', 'radha-krishna themes', 'communal dance'],
#     'Sun Temple Konark': ['temple', 'odisha', 'sun god', 'sculpture', 'architecture', 'chariot shape', 'Kalinga architecture', 'world heritage', 'astronomy'],
#     'iron_pillar': ['pillar', 'delhi', 'qutub minar', 'iron', 'history', 'corrosion resistance', 'metallurgical curiosity', 'Sanskrit inscriptions', 'gupta empire'],
#     'alai_darwaza': ['architecture', 'delhi', 'monument', 'gateway', 'mughal', 'entrance', 'Qutub complex', 'red sandstone', 'marble decorations'],
#     'Chhota_Imambara': ['mosque', 'lucknow', 'nawab', 'architecture', 'history', 'Shia Muslims', 'congregation hall', 'festivals', 'illumination'],
#     'charminar': ['mosque', 'hyderabad', 'monument', 'architecture', 'landmark', 'four minarets', 'global icon', 'Indo-Islamic architecture', 'city symbol'],
#     'Kangra': ['fort', 'himachal pradesh', 'kangra valley', 'history', 'architecture', 'ancient fort', 'earthquake ruins', 'strategic battles', 'maharaja sansar chand'],
#     'Hawa mahal': ['architecture', 'jaipur', 'rajasthan', 'pink city', 'palace', 'high screen wall', 'rajputana design', 'honeycomb', 'breeze structure'],
#     'Ellora Caves': ['cave', 'maharashtra', 'unesco', 'sculpture', 'architecture', 'rock-cut temples', 'Buddhist', 'Hindu', 'Jain', 'monolithic'],
#     'tanjavur temple': ['temple', 'tamil nadu', 'architecture', 'chola', 'unesco', 'brihadisvara temple', 'great living chola temples', 'dravidian architecture', 'vimanam'],
#     'Humayun_s Tomb': ['tomb', 'delhi', 'mughal', 'architecture', 'unesco', 'garden tomb', 'persian influence', 'prototype for Taj Mahal', 'indian subcontinent'],
#     'mysore_palace': ['palace', 'mysore', 'karnataka', 'architecture', 'royalty', 'Wodeyar dynasty', 'dussehra', 'indo-saracenic', 'chamundi hills'],
#     'India gate pics': ['war memorial', 'delhi', 'architecture', 'india', 'history', 'Amar Jawan Jyoti', 'WWI memorial', 'rajpath', 'an eternal flame','national monument', 'all india war memorial', 'patriotism'],
#     'Charar-E- Sharif': ['mosque', 'kashmir', 'sufi', 'architecture', 'history', 'wooden structure', 'muslim shrine', 'Sheikh Noor-ud-din', 'pilgrimage site'],
#     'tajmahal': ['mausoleum', 'agra', 'mughal', 'architecture', 'unesco', 'ivory-white marble', 'Shah Jahan', 'Mumtaz Mahal', 'symbol of love'],
#     'victoria memorial': ['memorial', 'kolkata', 'architecture', 'history', 'british', 'museum', 'queen victoria', 'british raj', 'landmark'],
#     'Ajanta Caves': ['cave', 'maharashtra', 'unesco', 'sculpture', 'architecture', 'buddhist art', 'rock-cut', 'murals', 'ancient'],
#     'kathak': ['dance', 'uttar pradesh', 'classical', 'performance', 'tradition', 'storytelling', 'ghungroos', 'spins', 'expressions'],
#     'Mural': ['art', 'painting', 'wall', 'mural', 'artist', 'public art', 'fresco', 'large scale', 'visual storytelling'],
#     'qutub_minar': ['minaret', 'delhi', 'monument', 'history', 'architecture', 'victory tower', 'islamic calligraphy', 'red sandstone', 'world heritage'],
#     'jamali_kamali_tomb': ['tomb', 'delhi', 'architecture', 'history', 'jamali kamali', 'mosque', 'mughal era', 'poet and sage', 'sufi saints']
# }

keywords_new= {

    'kathak': ['indian classical dance', 'storytelling', 'ghungroos', 'north india', 'nritta', 'nritya', 'abhinaya', 'footwork', 'spins', 'rhythmic patterns', 'expressive mime', 'technical purity', 'bells', 'dynamic flow', 'thumri', 'ghazals', 'bhajans', 'tarana', 'costumes', 'heritage'],
    'kuchipudi': ['indian classical dance', 'andhra pradesh', 'vocal narrative', 'expressive eyes', 'quick footwork', 'lively movements', 'drama', 'vaachika abhinaya', 'tarangam', 'jatis', 'bhama kalapam', 'siddhendra yogi', 'vedic chants', 'carved idols', 'temporal beauty', 'silk sarees', 'tradition', 'performance', 'mythological tales', 'religious themes'],
    'bharatanatyam': ['indian classical dance', 'tamil nadu', 'mudras', 'spiritual expression', 'temple tradition', 'carnatic music', 'alankar', 'precision', 'pure dance', 'solo performance', 'karanas', 'padams', 'javalis', 'bent knees', 'natya shastra', 'devadasis', 'spirituality', 'rigorous training', 'facial expressions', 'rhythmic accuracy'],
    'kathakali': ['indian classical dance', 'kerala', 'kathakali', 'vibrant makeup', 'elaborate costumes', 'gesture language', 'ritualistic', 'story play', 'facial expressions', 'kalaripayattu', 'eye movements', 'rama', 'mahabharata', 'sita', 'bhima', 'hand gestures', 'percussion', 'folk tales', 'cultural symbol', 'classical epics'],
    'odissi': ['indian classical dance', 'orissa', 'tribhangi', 'devotional', 'spiritual', 'sculpturesque', 'odissi music', 'batu nrutya', 'mangalacharan', 'pallavi', 'abhinaya', 'expressive prowess', 'gotipua', 'mahari tradition', 'sanskrit hymns', 'lyrical grace', 'historical evolution', 'dedicated schools', 'costume jewelry', 'cultural preservation'],
    'mohiniyattam': ['indian classical dance', 'kerala', 'femininity', 'lasya', 'sensuous', 'lyrical movements', 'solo performance', 'elegant costumes', 'sway of the body', 'vishnu', 'mohini', 'subtle abhinaya', 'gentle footwork', 'ghazals', 'bhajans', 'padams', 'carnatic ragas', 'expressive glances', 'dancers persona', 'graceful poise'],
    'sattriya': ['indian classical dance', 'assam', 'vaishnavite monasteries', 'devotional', 'drama dance', 'ankia naat', 'ritualistic', 'bhakti movement', 'ensemble', 'percussive rhythm', 'spiritual storytelling', 'xatriya tradition', 'classical purity', 'monastic origin', 'ornate costumes', 'borgeet', 'folk elements', 'vibrant theatre', 'historical plays', 'ritual dance'],
    'manipuri': ['indian classical dance', 'manipur', 'love stories', 'radha-krishna', 'gentle fluidity', 'rounded movements', 'devotional', 'tandava', 'lasya', 'raslila', 'pankhida', 'thougal jagoi', 'drum music', 'tribal influences', 'sankirtana', 'nupa pala', 'silk attire', 'dhol cholom', 'poetic charm', 'cultural festivals'],

    'Warli': ['folk art','Warli', 'maharashtra', 'tribal culture', 'ritual paintings', 'daily life', 'nature', 'geometric patterns', 'white pigment', 'earthly tones', 'murals', 'cultural storytelling', 'minimalistic design', 'harvest scenes', 'tribal dance', 'sacred art', 'animistic beliefs', 'ecological harmony', 'symbolic representation', 'traditional motifs', 'community life'],
    'Kangra': ['pahari painting','Kangra', 'kangra valley', 'radha-krishna', 'romantic themes', 'nature', 'miniatures', 'gita govinda', 'soft colors', 'detailed brushwork', 'courtly love', 'mythology', '16th century', 'delicate grace', 'floral motifs', 'mughal influence', 'spiritual romance', 'verdant landscapes', 'visual storytelling', 'traditional techniques', 'cultural heritage'],
    'Kalamkari': ['hand-painted','Kalamkari', 'block-printed', 'andhra pradesh', 'natural dyes', 'mythological narratives', 'organic dyes', 'cotton fabric', 'ancient craft', 'srikalahasti style', 'vegetable dyes', 'tree of life', 'folk tales', 'storytelling cloth', 'durable pigments', 'indian epics', 'religious tapestries', 'artisans', 'painting with a pen', 'intricate design', 'cultural symbolism'],
    'pattachitra': ['orissa','pattachitra', 'traditional cloth', 'hand-painted', 'icon painting', 'hindu mythology', 'ritual art', 'detailed imagery', 'vibrant colors', 'canvas work', 'folk art', 'temple murals', 'chariots of deities', 'palm leaves', 'religious processions', 'craftsmanship', 'narrative visuals', 'cultural traditions', 'gods and goddesses', 'artisan communities', 'historical art form'],
    'Mural': ['wall paintings', 'Mural','kerala', 'large scale', 'temple art', 'frescoes', 'historical depiction', 'religious motifs', 'public art', 'ancient technique', 'cultural narratives', 'architectural context', 'spiritual themes', 'mythological tales', 'colorful imagery', 'palace decorations', 'vast canvases', 'royal patronage', 'cave paintings', 'community engagement', 'artistic legacy'],
    'Portrait': ['personal likeness','Portrait', 'oil painting', 'character study', 'emotional depth', 'visual biography', 'detailed rendition', 'human subject', 'realism', 'expression capture', 'artistic representation', 'pictorial narrative', 'facial expression', 'classic portraiture', 'aesthetic detail', 'individuality', 'artistic interpretation', 'cultural identity', 'symbolic elements', 'visual storytelling', 'commissioned artwork'],
    'Tanjore': ['tamil nadu', 'gold leaf','Tanjore', 'glass beads', 'embossed art', 'religious themes', 'hindu gods', 'vivid colors', 'royal heritage', 'traditional canvas', 'rich ornamentation', 'precious stones', 'sacred artwork', 'dravidian architecture', 'spiritual iconography', 'luminous aura', 'cultural depiction', 'decorative motifs', 'divine portrayal', 'heritage craft', 'gallery display'],
    'Madhubani': ['bihar', 'mithila painting','Madhubani', 'folk art', 'natural pigments', 'geometric patterns', 'cultural expressions', 'bright colors', 'wedding rituals', 'floral animals', 'epic lore', 'tribal art', 'social occasions', 'fish and peacocks', 'village artisans', 'symbolic meanings', 'cultural motifs', 'environmental themes', 'handmade paper', 'traditional narratives', 'vibrant ecosystem'],
    
    'tajmahal': ['agra', 'mughal architecture', 'ivory-white marble', 'symbol of love', 'shah jahan', 'mumtaz mahal', 'persian influence', 'mausoleum', 'unesco world heritage', 'yamuna river', 'mughal empire', 'architectural marvel', 'eternal love', 'white domes', 'charbagh garden', 'historical monument', 'architectural symmetry', 'floral motifs', 'calligraphy art', 'cultural icon'],
    'lotus_temple': ['bahai faith', 'new delhi', 'lotus flower', 'modern architecture', 'peaceful worship', 'unity of religion', 'house of worship', 'spiritual symbol', 'lotus petals', 'prayer hall', 'community gathering', 'open to all', 'religious harmony', 'architectural wonder', 'lotus pond', 'meditative space', 'interfaith dialogue', 'spiritual unity', 'environmental design', 'lotus sculpture'],
    'golden temple': ['sikhism', 'amritsar', 'punjab', 'harmandir sahib', 'sikh gurus', 'holy pool', 'guru granth sahib', 'sikh community', 'spiritual pilgrimage', 'divine music', 'langar seva', 'sikh heritage', 'sikh architecture', 'sikh culture', 'religious tolerance', 'sikh history', 'sikh identity', 'sikh principles', 'sikh tradition', 'sikh values'],
    'charminar': ['hyderabad', 'telangana', 'indo-islamic architecture', 'landmark monument', 'four minarets', 'global icon', 'mosque structure', 'historical symbol', 'city of pearls', 'city symbol', 'islamic heritage', 'grand entrance', 'architectural marvel', 'historical legacy', 'cultural heritage', 'tourist attraction', 'cityscape view', 'marketplace', 'historical site', 'cultural symbol'],
    'India gate pics': ['new delhi', 'war memorial', 'india gate', 'martyrs memorial', 'amar jawan jyoti', 'eternal flame', 'world war i', 'victory arch', 'rajpath avenue', 'national monument', 'all india war memorial', 'indian army', 'indian soldiers', 'patriotic symbol', 'military sacrifice', 'martyrs tribute', 'national pride', 'historical landmark', 'architectural marvel', 'delhi tourism'],
    'Gateway of India': ['mumbai', 'india gate', 'mumbai harbor', 'basalt arch', '20th-century', 'british raj', 'historical monument', 'indian independence', 'victory arch', 'rajpath avenue', 'landmark structure', 'tourist attraction', 'arabian sea', 'historical gateway', 'british architecture', 'colonial legacy', 'indian heritage', 'gateway to india', 'indian tourism', 'historical site'],
    'victoria memorial': ['kolkata', 'british raj', 'queen victoria', 'memorial museum', 'british architecture', 'historical landmark', 'colonial heritage', 'victorian era', 'british monarchy', 'royal memorial', 'british empire', 'historical monument', 'landmark structure', 'indian history', 'colonial legacy', 'british rule', 'indian heritage', 'historical site', 'cultural symbol', 'victorian art'],
    'Fatehpur Sikri': ['agra', 'mughal architecture', 'unesco world heritage', 'red sandstone', 'akbar the great', 'buland darwaza', 'fatehpur sikri fort', 'mughal empire', 'historical site', 'mughal history', 'mughal era', 'mughal emperor', 'mughal capital', 'mughal city', 'mughal palace', 'mughal legacy', 'mughal culture', 'mughal art', 'mughal design', 'mughal style'],
    'Ellora Caves': ['rock-cut architecture', 'aurangabad', 'unesco world heritage site', 'buddhist', 'hindu', 'jain temples', 'cave complex', 'kailasa temple', 'religious art', 'sculptural panels', 'ancient monasteries', 'bas-relief', 'indian rock-cut architecture', 'indological study', 'pilgrimage site', 'archaeological heritage', 'medieval period', 'cultural synthesis', 'monolithic shrines', 'viswakarma cave'],
    'Ajanta Caves': ['buddhist rock-cut caves', 'aurangabad', 'unesco world heritage site', 'ancient frescoes', 'wall paintings', 'jataka tales', 'bodhisattva', 'monastic life', 'archaeological art', 'historic manuscripts', 'indian mural tradition', 'buddhist heritage', 'silk route', 'cave sanctuaries', 'iconography', 'religious devotion', 'cave temple', 'early buddhist architecture', 'tourist attraction', 'archaeological conservation'],
    'Sun Temple Konark': ['odisha', 'konark sun temple', 'surya temple', 'unesco world heritage site', 'chariot temple', 'kalinga architecture', 'sun god', 'surya deva', 'astronomical significance', 'archaeological marvel', 'historical monument', 'hindu temple', 'religious architecture', 'stone carvings', 'architectural grandeur', 'historical site', 'cultural heritage', 'archaeological site', 'sun worship', 'konark beach'],
    'mysore_palace': ['karnataka', 'mysore', 'royal palace', 'wodeyar dynasty', 'indo-saracenic architecture', 'dussehra festival', 'royal residence', 'chamundi hills', 'palace architecture', 'royal heritage', 'cultural landmark', 'historical monument', 'royal family', 'royal history', 'royal patronage', 'royal architecture', 'royal lifestyle', 'royal culture', 'royal tradition', 'royal legacy'],
    'Humayun_s Tomb': ['delhi', 'mughal architecture', 'unesco world heritage site', 'garden tomb', 'persian influence', 'prototype for taj mahal', 'mughal emperor', 'mughal history', 'mughal era', 'mughal capital', 'mughal legacy', 'mughal culture', 'mughal art', 'mughal design', 'mughal style', 'mughal garden', 'mughal emperor', 'mughal architecture', 'mughal history', 'mughal era'],
    'Chhota_Imambara': ['lucknow', 'nawab', 'shia muslims', 'imambara mosque', 'chhota imambara', 'nawabi architecture', 'nawabi culture', 'nawabi heritage', 'nawabi style', 'nawabi history', 'nawabi era', 'nawabi legacy', 'nawabi tradition', 'nawabi palace', 'nawabi lifestyle', 'nawabi city', 'nawabi cuisine', 'nawabi cuisine', 'nawabi art', 'nawabi design'],
    'jamali kamali tomb': ['archaeological site', 'mehrauli', 'delhi', 'sufi saints', 'lodi era', 'sandalwood doors', 'red sandstone', 'marble inlays', 'poetry inscriptions', 'mosque and tomb', 'historical monument', 'mughal architecture', 'ornate design', 'blue tiles', 'calligraphy', 'heritage walk', 'jamali', 'kamali', 'restored site', 'indian heritage'],
    'qutub minar': ['world heritage site', 'minaret', 'delhi', 'islamic monument', 'mughal architecture', 'marble inscriptions', 'quwwat-ul-islam mosque', 'iron pillar', 'sandstone fluting', 'balconies', 'historical landmark', 'victory tower', 'intricate carvings', 'aibak', 'mehrauli', 'tapering tower', 'architectural innovation', 'cultural significance', 'ancient scripts', 'tourist attraction'],
    'tanjavur temple': ['brihadeeswarar temple', 'tamil nadu', 'unesco world heritage site', 'chola dynasty', 'dravidian architecture', 'great living chola temples', '1010 ce', 'rajendra chola i', 'vast compound', 'nandi bull', 'gopuram', 'granite carvings', 'sacred complex', 'shaivism', 'indian temples', 'southern india', 'architectural marvel', 'cultural icon', 'religious pilgrimage', 'ancient engineering'],
    'alai_darwaza': ['gateway', 'qutub complex', 'delhi', 'alauddin khalji', 'red sandstone', 'islamic architecture', 'decorative motifs', 'mughal art', 'horseshoe arch', 'blue tiles', 'inscription panels', 'indian monuments', '13th century', 'indo-islamic fusion', 'domed entrance', 'architectural ornamentation', 'heritage site', 'restoration work', 'cultural syncretism', 'medieval india'],
    'iron_pillar': ['mehrauli', 'corrosion resistance', 'metallurgical curiosity', 'gupta empire', 'sanscrit inscription', 'iron metallurgy', 'qutub complex', 'chandragupta ii', 'rust proof', 'ancient craftsmanship', 'historical enigma', 'monolithic structure', 'traditional smithing', 'cultural significance', 'archaeological study', 'pillar cult', 'ancient technology', 'heritage conservation', 'indological research', 'tourist landmark'],
    'hawa mahal pics': ['palace of winds', 'jaipur', 'rajasthan', 'pink city', 'rajput architecture', 'red and pink sandstone', 'royal ladies', 'privacy screen', '953 windows', 'lattice work', 'five storeys', 'cultural icon', 'indian palaces', 'maharaja sawai pratap singh', '1799 ce', 'vedic architecture', 'iconic facade', 'historical architecture', 'royal heritage', 'tourist attraction'],
    'alai_minar': ['unfinished minaret', 'alauddin khalji', 'qutub complex', 'delhi', 'red sandstone', 'mughal architecture', 'double the height of qutub minar', 'khalji dynasty', 'historical site', 'islamic monument', 'archaeological remains', 'megalithic structure', 'medieval india', 'ambitious project', 'ruins', 'cultural heritage', 'tourist spot', 'architectural history', 'imperial vision', 'monumental tower'],
    'basilica_of_bom_jesus': ['goa', 'roman catholic basilica', 'baroque architecture', 'unesco world heritage site', 'st. francis xavier', 'relics', 'sacred art', 'jesuit church', 'indian christianity', 'spiritual site', 'portuguese india', '17th century', 'pilgrimage site', 'religious monument', 'architectural grandeur', 'cultural intermix', 'heritage preservation', 'western ghats', 'tourism landmark', 'ecclesiastical history'],
    'Khajuraho': ['madhya pradesh', 'temple complex', 'chandela dynasty', 'nagara-style architectural symbolism', 'unesco world heritage site', 'erotic sculptures', 'jain temples', 'hinduism', 'medieval temple art', 'kamasutra reliefs', 'architectural brilliance', 'cultural astronomy', 'ancient erotica', 'granite and sandstone', 'symmetry in architecture', 'intricate carvings', 'cultural evolution', 'mythological depictions', 'pinnacle of indian architecture', 'indological studies'],
    'Charar-E- Sharif': ['kashmir', 'sufi shrine', 'muslim pilgrimage', 'wooden architecture', 'islamic heritage', 'sufi saints', 'religious harmony', 'historical monument', 'religious tolerance', 'cultural icon', 'spiritual site', 'muslim architecture', 'sufi culture', 'sufi tradition', 'sufi music', 'sufi poetry', 'sufi mysticism', 'sufi teachings', 'sufi practices', 'sufi rituals'],

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
                    if label in keywords_new:
                        keywords = ', '.join(keywords_new[label])
                        output_row = [Class, label, cluster, most_similar_image_path, keywords]
                        
                        with open(output_csv_path, mode='a' if file_exists else 'w', newline='', encoding='utf-8') as file:
                            writer = csv.writer(file)
                            if not file_exists:
                                writer.writerow(['Class', 'Label', 'Cluster', 'Most_Similar_Image_Path', 'Key_Words'])
                                file_exists = True
                            writer.writerow(output_row)

print("All files have been processed and results appended to:", output_csv_path)