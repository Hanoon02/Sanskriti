from flask import Flask, render_template, request
import os
import pickle
from transformers import pipeline
from bs4 import BeautifulSoup
import requests

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

def fetch_website_content(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return ' '.join([p.get_text() for p in soup.find_all('p')])
    except Exception as e:
        print(f"Error fetching website content: {e}")
    return ""

def generate_wikipedia_url(search_query):
    wikipedia_api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'search',
        'srsearch': search_query
    }
    try:
        response = requests.get(wikipedia_api_url, params=params)
        response.raise_for_status()
        data = response.json()
        page_id = data['query']['search'][0]['pageid']
        return f"https://en.wikipedia.org/wiki?curid={page_id}"
    except Exception as e:
        print(f"Error fetching Wikipedia URL: {e}")
    return None

wikipedia_url = generate_wikipedia_url('Indian Art')
content = fetch_website_content(wikipedia_url)
def answer_question(question, context):
    qa_pipeline = pipeline('question-answering', model='twmkn9/bert-base-uncased-squad2')
    return qa_pipeline(question=question, context=context)['answer']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    input_data = request.form['input_data']
    output = answer_question(input_data,content)
    return render_template('result.html', output=output)

if __name__ == '__main__':
    app.run(debug=True, extra_files=extra_files)
