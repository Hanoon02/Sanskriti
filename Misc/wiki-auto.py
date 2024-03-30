import os
import wikipedia
from tqdm import tqdm

def download_articles(query, hindi=False):
    data_directory = "AutoRawData"
    query_directory = os.path.join(data_directory, query)
    if not os.path.exists(query_directory):
        os.makedirs(query_directory)
    
    if hindi:
        wikipedia.set_lang("hi")
    else:
        wikipedia.set_lang("en")

    fetched_articles = 0
    try:
        search_results = wikipedia.search(query, results=100000000000)
        search_results = [result for result in search_results if "(disambiguation)" not in result]

        for selected_result in tqdm(search_results, desc="Downloading articles"):
            file_path = os.path.join(query_directory, f"{selected_result}.txt")
            if os.path.exists(file_path):
                continue
            try:
                search = wikipedia.page(selected_result).content
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(search)
                fetched_articles += 1
                if fetched_articles >= len(search_results):
                    break
            except Exception as e:
                print(f"An error occurred while processing {selected_result}")

        if fetched_articles >= len(search_results):
            print(f"Successfully downloaded {fetched_articles} articles for query '{query}'.")
        else:
            print(f"Could not fetch enough articles for query '{query}'.")
    except Exception as e:
        print(f"An unexpected error occurred")

try:
    query = input("Enter your Wikipedia query: ")
    hindi = query.endswith('h')
    download_articles(query, hindi)
except ValueError as e:
    print("Invalid Input: Please enter a valid number.")
except Exception as e:
    print("An unexpected error occurred:")
