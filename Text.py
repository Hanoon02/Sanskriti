import requests
from bs4 import BeautifulSoup
from transformers import pipeline

def fetch_wikipedia_content(url):
    user_agent = "Sanskriti/1.0 (himani21053@iiitd.ac.in)"  # Replace with your actual email
    headers = {'User-Agent': user_agent}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        content = ' '.join([p.get_text() for p in soup.find_all('p')])
        return content

    except requests.exceptions.HTTPError as errh:
        print("HTTP Error:", errh)
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting:", errc)
    except requests.exceptions.Timeout as errt:
        print("Timeout Error:", errt)
    except requests.exceptions.RequestException as err:
        print("Error:", err)

    return None

def answer_question(question, context):
    qa_pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2')
    result = qa_pipeline(question=question, context=context)
    answer = result['answer']

    if answer:
        return answer
    else:
        return "No answer found."

def main():
    topics = [
        'https://en.wikipedia.org/wiki/Indian_classical_dance',
        'https://en.wikipedia.org/wiki/Indian_art',
        'https://en.wikipedia.org/wiki/List_of_World_Heritage_Sites_in_India'
    ]

    for topic in topics:
        print(f"\nFetching Wikipedia data for {topic}")
        content = fetch_wikipedia_content(topic)

        if content:
            user_question = input(f"Ask a question of our Sanskriti: ")
            answer = answer_question(user_question, content)
            print(f"Answer: {answer}\n")
        else:
            print(f"No Wikipedia data found for {topic}")

if __name__ == "__main__":
    main()
