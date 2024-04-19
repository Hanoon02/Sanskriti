import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from cachetools import TTLCache, cached
from bert_score import score as bert_score
from transformers import MarianMTModel, MarianTokenizer


nltk.download('punkt')

cache = TTLCache(maxsize=100, ttl=600)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

@cached(cache)
def fetch_website_content(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content = ' '.join([p.get_text() for p in soup.find_all('p')])
        return preprocess_text(content)
    except Exception as e:
        print(f"Error fetching website content: {e}")
    return ""

def generate_wikipedia_url(search_query, lang="en"):
    wikipedia_api_url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'search',
        'srsearch': search_query
    }
    try:
        response = requests.get(wikipedia_api_url, params=params)
        data = response.json()
        page_id = data['query']['search'][0]['pageid']
        return f"https://{lang}.wikipedia.org/wiki?curid={page_id}"
    except Exception as e:
        print(f"Error fetching Wikipedia URL in {lang}: {e}")
    return None

def answer_question_in_english(question, context):
    model_name = 'twmkn9/bert-base-uncased-squad2'
    qa_pipeline = pipeline('question-answering', model=model_name)
    return qa_pipeline(question=question, context=context)['answer']

def translate_answer_to_hindi(answer_english):
    model_name = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translated = model.generate(**tokenizer([answer_english], return_tensors="pt", padding=True, truncation=True, max_length=512))
    hindi_answer = tokenizer.decode(translated[0], skip_special_tokens=True)
    return hindi_answer

def main():
    test_file_path = "input.json"
    with open(test_file_path, 'r', encoding='utf-8') as file:
        test_questions = json.load(file)
    
    for item in test_questions:
        topic = item['topic']
        wikipedia_url = generate_wikipedia_url(topic)
        if not wikipedia_url:
            print(f"Could not retrieve URL for the topic: {topic}")
            continue
        
        content = fetch_website_content(wikipedia_url)
        for question in item['questions']:
            answer_english = answer_question_in_english(question, content)
            answer_hindi = translate_answer_to_hindi(answer_english)
            
            print(f"Topic: {topic}")
            print(f"Question: {question}")
            print(f"Predicted Answer in English: {answer_english}")
            print(f"Predicted Answer in Hindi: {answer_hindi}\n---")

if __name__ == "__main__":
    main()
