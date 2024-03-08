import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import json
from bert_score import score as bert_score
import string

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

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def answer_question(question, context):
    qa_pipeline = pipeline('question-answering', model='twmkn9/bert-base-uncased-squad2')
    return qa_pipeline(question=question, context=context)['answer']

def evaluate_with_bert_score(prediction, reference):
    _, _, F1 = bert_score([prediction], [reference], lang="en", verbose=True)
    return F1.mean().item()

def load_reference_answers(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def find_reference_answer(question, reference_answers):
    question = preprocess_text(question)
    for item in reference_answers:
        if preprocess_text(item['question']) == question:
            return item['answers'][0]['text']
    return None

def save_output(data, output_file_path):
    with open(output_file_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(data) + "\n")

def read_outputs_and_calculate_map(output_file_path):
    total_score, count = 0, 0
    with open(output_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            total_score += data['bert_score_f1']
            count += 1
    average_score = total_score / count if count else 0
    print(f"Mean of BERT F1 scores (Simplified MAP): {average_score}")

def process_test_questions(test_file_path, reference_file_path, output_file_path):
    test_questions = load_reference_answers(test_file_path)  
    reference_answers = load_reference_answers(reference_file_path)
    
    for item in test_questions:
        topic = item['topic']
        wikipedia_url = generate_wikipedia_url(topic)
        if not wikipedia_url:
            print(f"Could not retrieve URL for the topic: {topic}")
            continue
        content = fetch_website_content(wikipedia_url)
        if not content:
            print(f"No content found for the topic: {topic}")
            continue
        
        for question in item['questions']:
            answer = answer_question(question, content)
            reference_answer = find_reference_answer(question, reference_answers)
            if reference_answer:
                f1_score = evaluate_with_bert_score(answer, reference_answer)
                save_output({'topic': topic, 'question': question, 'predicted_answer': answer, 'reference_answer': reference_answer, 'bert_score_f1': f1_score}, output_file_path)
                print(f"Topic: {topic}\nQuestion: {question}\nAnswer: {answer}\nReference Answer: {reference_answer}\nBERT Score F1: {f1_score}\n---")
            else:
                print(f"Reference answer not found for question: {question}")

def main():
    test_file_path = "Sanskriti//input.json"
    reference_file_path = "qa_IDC.json"
    output_file_path = "model_predictions_and_references.json"
    process_test_questions(test_file_path, reference_file_path, output_file_path)
    read_outputs_and_calculate_map(output_file_path)

if __name__ == "__main__":
    main()
