from transformers import pipeline
from bs4 import BeautifulSoup
import requests

class BertModelText:
    def __init__(self, input):
        self.input = input
        self.wikipedia_url = self.generate_wikipedia_url('Indian Art')
        self.content = self.fetch_website_content(self.wikipedia_url)
        
    def fetch_website_content(self, url):
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            return ' '.join([p.get_text() for p in soup.find_all('p')])
        except Exception as e:
            print(f"Error fetching website content: {e}")
        return ""

    def generate_wikipedia_url(self, search_query):
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

    def answer_question(self, question, context):
        qa_pipeline = pipeline('question-answering', model='twmkn9/bert-base-uncased-squad2')
        return qa_pipeline(question=question, context=context)['answer']
    
    def compute(self):
        return self.answer_question(self.input, self.content)