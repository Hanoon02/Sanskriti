import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from textstat.textstat import textstatistics

def load_data_from_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def analyze_data(data):
    num_questions = len(data)
    avg_question_length = sum(len(item["question"]) for item in data) / num_questions
    avg_answer_length = sum(len(item["answers"][0]["text"]) for item in data) / num_questions

    # Content Analysis
    all_questions = " ".join(item["question"] for item in data)
    all_answers = " ".join(item["answers"][0]["text"] for item in data)

    # NLP Analysis - Named Entity Recognition (NER)
    entities = []
    nlp = spacy.load("en_core_web_sm")
    for item in data:
        doc = nlp(item["answers"][0]["text"])
        entities.extend([(ent.text, ent.label_) for ent in doc.ents])

    # Readability Scores
    readability_scores = [textstatistics().flesch_reading_ease(item["answers"][0]["text"]) for item in data]

    return num_questions, avg_question_length, avg_answer_length, all_questions, all_answers, entities, readability_scores

def generate_wordcloud(text, ngram_range=(1, 1)):
    count_vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    count_data = count_vectorizer.fit_transform([text])
    words = count_vectorizer.get_feature_names_out()
    total_counts = count_data.sum(axis=0)
    count_dict = (zip(words, total_counts.tolist()[0]))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)
    words_dict = dict(count_dict)
    return WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words_dict)

def visualize_data(wordcloud_questions, wordcloud_answers, wordcloud_bi_grams):
    # Visualizations
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_questions, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud for Questions")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_answers, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud for Answers")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_bi_grams, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud for Bi-grams")
    plt.show()

def print_analysis_results(num_questions, avg_question_length, avg_answer_length, entities, readability_scores):
    print(f"Number of questions: {num_questions}")
    print(f"Average question length: {avg_question_length} characters")
    print(f"Average answer length: {avg_answer_length} characters")
    print(f"Named Entities (Sample): {entities[:10]}")
    print(f"Sample Readability Scores: {readability_scores}")

def main():
    file_path = "qa_idc.json"
    data = load_data_from_json(file_path)
    num_questions, avg_question_length, avg_answer_length, all_questions, all_answers, entities, readability_scores = analyze_data(data)
    
    wordcloud_questions = generate_wordcloud(all_questions)
    wordcloud_answers = generate_wordcloud(all_answers)
    wordcloud_bi_grams = generate_wordcloud(all_questions + ' ' + all_answers, ngram_range=(2, 2))
    
    visualize_data(wordcloud_questions, wordcloud_answers, wordcloud_bi_grams)
    
    print_analysis_results(num_questions, avg_question_length, avg_answer_length, entities, readability_scores)

if __name__ == "__main__":
    main()
