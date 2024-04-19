import requests
from bs4 import BeautifulSoup
import spacy
import json
from transformers import pipeline

nlp = spacy.load("en_core_web_sm")

def fetch_wikipedia_content(url):
    user_agent = "Sanskriti/1.0 (example@example.com)"
    headers = {'User-Agent': user_agent}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    content = ' '.join([p.get_text() for p in soup.find_all('p') if len(p.get_text()) > 40])
    return content

def refine_answer(doc, ent):
    sent = next((sent for sent in doc.sents if ent.start_char >= sent.start_char and ent.end_char <= sent.end_char), None)
    if not sent:
        return None, None  
    start_token = max(ent.start - 8, sent.start)  
    end_token = min(ent.end + 8, sent.end)  
    refined_answer = doc[start_token:end_token].text
    return refined_answer, sent.start_char

def generate_qa_pairs(text, qa_pipeline):
    doc = nlp(text)
    qa_pairs = []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "NORP", "GPE", "PERSON", "LOC"]:  
            question = f"what is {ent.text}famous for?"
            answer = qa_pipeline(question=question, context=text)
            if answer and answer['score'] > 0.1:  
                qa_pairs.append({
                    "context": text,
                    "question": question,
                    "answers": [{"text": answer['answer'], "answer_start": answer['start']}],
                    "is_impossible": False
                })
    return qa_pairs

def save_qa_pairs(qa_pairs, filename="qa_IA_2.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=4, ensure_ascii=False)

def main():
    # Initialize the QA pipeline
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    
    # Fetch content
    url = 'https://en.wikipedia.org/wiki/Indian_art'
    content = fetch_wikipedia_content(url)
    
    # Generate Q&A pairs
    qa_pairs = generate_qa_pairs(content, qa_pipeline)
    
    # Save to file
    save_qa_pairs(qa_pairs)
    print(f"Saved {len(qa_pairs)} Q&A pairs to 'qa_pairs.json'")

if __name__ == "__main__":
    main()