import os
import wikipedia

try:
    query = input("Enter your Wikipedia query: ")
    hindi = False
    if query.endswith('h'):
        hindi = True
    search_results = wikipedia.search(query, results=5)
    print("Search Results:")
    for i, result in enumerate(search_results, start=1):
        print(f"{i}. {result}")

    selection = int(input("Enter the number corresponding to the desired search result: "))
    selected_result = search_results[selection - 1]
    if hindi:
        wikipedia.set_lang("hi")  
    search = wikipedia.page(selected_result).content
    data_directory = "RawData"
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    file_path = os.path.join(data_directory, f"{selected_result}.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(search)

    print("Data saved successfully.")
except wikipedia.exceptions.DisambiguationError as e:
    print("Disambiguation Error: There are multiple possible pages for this query. Please try again with a more specific query.")
except wikipedia.exceptions.PageError as e:
    print("Page Error: The requested page does not exist. Please try again with a different query or check for typos.")
except (ValueError, IndexError) as e:
    print("Invalid Input: Please enter a valid number corresponding to the desired search result.")
except Exception as e:
    print("An unexpected error occurred:", e)
