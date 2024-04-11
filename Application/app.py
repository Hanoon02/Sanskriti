from flask import Flask, render_template, request
import os
import pickle
from time import time
from text import BertModelText
from image import MonumentImage

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
    language = request.form.get('language', 'english')
    bertModelText = BertModelText(input_data)
    output = bertModelText.compute()
    image_file = request.files['image_input']
    unique_filename = f"{int(time())}.jpg" 
    image_path = os.path.join(app.config['SAVE_FOLDER'], unique_filename)
    image_file.save(image_path)
    monumentImage = MonumentImage(image_path)
    output_image = monumentImage.compute()
    fetch_path = os.path.join(app.config['FETCH_FOLDER'], unique_filename)
    return render_template('result.html', question= input_data, output=output, image_path=fetch_path, language = language, image_class = output_image)

@app.route('/clean_image')
def clean_image():
    try:
        upload_folder = app.config['SAVE_FOLDER']
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

app.config['SAVE_FOLDER'] = 'static/uploads'
app.config['FETCH_FOLDER'] = 'uploads'
app.run(debug=True, extra_files=extra_files)
