from flask import Flask, render_template, request, jsonify
import os
from time import time
from helper_code import ImageInputData, Chatbot, TextToImage
from shutil import copyfile
import requests

chatbot = Chatbot()

app = Flask(__name__)

extra_dirs = ['templates']
extra_files = extra_dirs[:]
for extra_dir in extra_dirs:
    for dirname, _, filenames in os.walk(extra_dir):
        for filename in filenames:
            filename = os.path.join(dirname, filename)
            if os.path.isfile(filename):
                extra_files.append(filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form['input_data']
    img_class = None
    img_label = None
    search_image_paths = []
    if chatbot.contains_image_keywords(input_data):
        textToImage = TextToImage()
        img_class, img_label = textToImage.find_most_relevant_label(input_data)
        paths = textToImage.fetch_img(img_class, img_label)
        for path in paths:
            filename = f"{int(time())}_{os.path.basename(path)}"
            output_image_path = os.path.join('static', 'image_outputs', filename)
            copyfile(path, output_image_path)
            search_image_paths.append(os.path.join('image_outputs', filename))
    language = request.form.get('language', 'english')
    try:
        image_file = request.files['image_input']
    except:
        image_file = None
    if image_file is None:  
        image_path = None
        output_image = ""
        similar_image_paths = []
    else:
        unique_filename = f"{int(time())}.jpg" 
        image_path = os.path.join(app.config['SAVE_FOLDER'], unique_filename)
        image_file.save(image_path)
        imgData = ImageInputData()
        output_image = imgData.get_class(image_path)
        similar_images = imgData.get_similiar_images(image_path)
        similar_image_paths = []
        for similar_image_path in similar_images:
            filename = f"{int(time())}_{os.path.basename(similar_image_path)}"
            output_image_path = os.path.join('static', 'image_outputs', filename)
            copyfile(similar_image_path, output_image_path)
            similar_image_paths.append(os.path.join('image_outputs', filename))
    
    fetch_path = os.path.join(app.config['FETCH_FOLDER'], unique_filename) if image_path else None
    return render_template('result.html', question=input_data, output='', image_path=fetch_path, language=language, image_class=output_image, similar_images=similar_image_paths, pred_class = img_class, pred_label = img_label, search_images = search_image_paths)


@app.route('/clean_image')
def clean_image():
    try:
        upload_folder = app.config['SAVE_FOLDER']
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        upload_folder = 'static/image_outputs'
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        return 'All images cleaned successfully!'
    except Exception as e:
        return f'Error cleaning images: {str(e)}'
    
@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/download-image', methods=['POST'])
def download_image():
    data = request.get_json()
    image_url = data.get('imageUrl')
    if not image_url:
        return jsonify({'error': 'Image URL not provided'}), 400
    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to download image'}), 500
        download_dir = 'static/downloads'
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        image_path = os.path.join(download_dir, 'downloaded_image.jpg')
        with open(image_path, 'wb') as f:
            f.write(response.content)
        return jsonify({'message': 'Image downloaded successfully', 'imagePath': image_path}), 200
    except Exception as e:
        return jsonify({'error': f'Error downloading image: {str(e)}'}), 500

app.config['SAVE_FOLDER'] = 'static/uploads'
app.config['FETCH_FOLDER'] = 'uploads'
app.run(debug=True, extra_files=extra_files)
