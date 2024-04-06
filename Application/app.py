from flask import Flask, render_template, request
import os
import pickle

app = Flask(__name__)

# Specify additional directories to be watched for changes
extra_dirs = ['templates']  # Add other directories as needed
extra_files = extra_dirs[:]
for extra_dir in extra_dirs:
    for dirname, _, filenames in os.walk(extra_dir):
        for filename in filenames:
            filename = os.path.join(dirname, filename)
            if os.path.isfile(filename):
                extra_files.append(filename)

# with open('models/trained_model.pkl', 'rb') as f:
#     model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    input_data = request.form['input_data']
    output = input_data
    return render_template('result.html', output=output)

if __name__ == '__main__':
    app.run(debug=True, extra_files=extra_files)
