import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import json
from bert_score import score as bert_score
import string
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy
import time

nltk.download('punkt')
nltk.download('wordnet')

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        return json.load(file)

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

def create_inverted_index(docs):
    inverted_index = {}
    for i, doc in enumerate(docs):
        for word in nltk.word_tokenize(doc):
            if word not in inverted_index:
                inverted_index[word] = []
            inverted_index[word].append(i)
    return inverted_index

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def rank_documents(query, docs, vectorizer):
    query_vec = vectorizer.transform([query])
    doc_vecs = vectorizer.transform(docs)
    cosine_similarities = cosine_similarity(query_vec, doc_vecs).flatten()
    ranked_docs = [docs[i] for i in cosine_similarities.argsort()[::-1]]
    return ranked_docs

def calculate_cosine_similarity(doc_vectors, query_vector):
    cosine_similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    return cosine_similarities

def calculate_cosine_similarity_for_each_pair(query_vector, pair_vectors):
    cosine_similarities = cosine_similarity(query_vector, pair_vectors)
    return cosine_similarities.flatten()

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

def process_test_questions_with_inverted_index(test_file_path, reference_file_path, output_file_path, vectorizer, inverted_index, doc_vectors):
    test_questions = load_reference_answers(test_file_path)  
    reference_answers = load_reference_answers(reference_file_path)
    
    # Vectorize each question-answer pair
    pair_vectors = vectorizer.transform([preprocess_text(pair['question'] + ' ' + pair['answers'][0]['text']) for pair in reference_answers])

    for item in test_questions:
        topic = item['topic']
        wikipedia_url = generate_wikipedia_url(topic)
        if not wikipedia_url:
            print(f"Could not retrieve URL for the topic: {topic}")
            continue
        
        # Measure start time
        start_time = time.time()
        
        content = fetch_website_content(wikipedia_url)
        
        # Measure end time
        end_time = time.time()
        
        if not content:
            print(f"No content found for the topic: {topic}")
            continue
        
        # Calculate duration
        duration = end_time - start_time
        print(f"Time taken to retrieve content for '{topic}': {duration} seconds")
        
        # Preprocess and vectorize the query
        preprocessed_query = preprocess_text(topic)
        query_vector = vectorizer.transform([preprocessed_query])
        
        # Calculate cosine similarity for each pair
        cosine_similarities = calculate_cosine_similarity_for_each_pair(query_vector, pair_vectors)
        
        # Retrieve relevant document IDs from the inverted index
        query_tokens = nltk.word_tokenize(preprocessed_query)
        relevant_doc_ids = set()
        for token in query_tokens:
            if token in inverted_index:
                relevant_doc_ids.update(inverted_index[token])
        
        # Retrieve relevant document vectors from the TF-IDF matrix
        relevant_doc_vectors = doc_vectors[list(relevant_doc_ids)]
        
        for question in item['questions']:
            answer = answer_question(question, content)
            reference_answer = find_reference_answer(question, reference_answers)
            if reference_answer:
                f1_score = evaluate_with_bert_score(answer, reference_answer)
                # Find the index of the most relevant question-answer pair
                most_relevant_pair_index = cosine_similarities.argmax()
                # Get the cosine similarity for the most relevant pair
                cosine_similarity_score = cosine_similarities[most_relevant_pair_index]
                
                save_output({
                    'topic': topic,
                    'question': question,
                    'predicted_answer': answer,
                    'reference_answer': reference_answer,
                    'bert_score_f1': f1_score,
                    'cosine_similarity': cosine_similarity_score,
                    'query_retrieval_time': duration  # Include query retrieval time in output
                }, output_file_path)
                
                print(f"Topic: {topic}\nQuestion: {question}\nAnswer: {answer}\nReference Answer: {reference_answer}\nBERT Score F1: {f1_score}\nCosine Similarity: {cosine_similarity_score}\n---")
            else:
                print(f"Reference answer not found for question: {question}")

    print("Results from the inverted index:")
    for word, postings in inverted_index.items():
        print(f"Word: {word}, Postings: {postings}")

                
def save_output(data, output_file_path):
    # Convert numpy int64 to Python int
    for key, value in data.items():
        if isinstance(value, numpy.int64):
            data[key] = int(value)
    
    with open(output_file_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(data) + "\n")

def main():
    test_file_path = "input.json"
    reference_file_path = "qa_IDC.json"
    output_file_path = "model_predictions_and_references.json"

    # Load the reference dataset
    with open(reference_file_path, 'r') as file:
        reference_answers = json.load(file)

    # Preprocess the documents
    preprocessed_docs = [preprocess_text(doc['context']) for doc in reference_answers]

    # Create the inverted index
    inverted_index = create_inverted_index(preprocessed_docs)

    # Initialize the vectorizer for the vector space model
    vectorizer = TfidfVectorizer()
    vectorizer.fit(preprocessed_docs)

    # Transform documents to vectors
    doc_vectors = vectorizer.transform(preprocessed_docs)

    # Process test questions using the inverted index
    process_test_questions_with_inverted_index(test_file_path, reference_file_path, output_file_path, vectorizer, inverted_index, doc_vectors)

if __name__ == "__main__":
    main()
