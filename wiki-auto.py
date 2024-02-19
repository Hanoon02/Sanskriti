import os
import wikipedia
from tqdm import tqdm 

try:
    query = input("Enter your Wikipedia query: ")
    hindi = False
    if query.endswith('h'):
        hindi = True  
    search_results = wikipedia.search(query, results=None)  
    max_selection = min(int(input("Enter the number of results to fetch: ")), len(search_results))
    selected_results = search_results[:max_selection]
    
    data_directory = "AutoRawData"
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    
    query_directory = os.path.join(data_directory, query)
    if not os.path.exists(query_directory):
        os.makedirs(query_directory)
    
    if hindi:
        wikipedia.set_lang("hi")  
    
    for selected_result in tqdm(selected_results, desc="Downloading articles"):
        try:
            search = wikipedia.page(selected_result).content
            file_path = os.path.join(query_directory, f"{selected_result}.txt")
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(search)
        except Exception as e:
            print(f"An error occurred while processing {selected_result}")

    print("Data saved successfully.")
except wikipedia.exceptions.DisambiguationError as e:
    print("Disambiguation Error: There are multiple possible pages for this query. Please try again with a more specific query.")
except wikipedia.exceptions.PageError as e:
    print("Page Error: The requested page does not exist. Please try again with a different query or check for typos.")
except (ValueError, IndexError) as e:
    print("Invalid Input: Please enter a valid number corresponding to the desired search result.")
except Exception as e:
    print("An unexpected error occurred:", e)
