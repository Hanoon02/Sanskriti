import pathlib
import textwrap
from pprint import pprint
import google.generativeai as genai
from PIL import Image
import time
import requests
from tqdm import tqdm

GOOGLE_API_KEY = "AIzaSyDBj0Oa1mF5Uk60deeGtlBclLqeV_Zfi6s"
genai.configure(api_key=GOOGLE_API_KEY)


model = genai.GenerativeModel("gemini-pro")


def model_input_1():
    prompt = f"""
    What is Indian Classical Dance"""
    return prompt


safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


def generate_answer_1():

    prompt = model_input_1()
    # print("not resolved")
    # response = model.generate_content(model_input_1(sample), safety_settings = safety_settings)
    response = model.generate_content(
        [prompt],
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
        ),
        safety_settings=safety_settings,
        stream=True,
    )
    # print("resolved_check")
    response.resolve()
    # print("resolved")
    # print(response.text)

    generated_text = ""

    if hasattr(response, "candidates") and response.candidates:
        # print(1)
        # Assuming we want the first candidate's first part
        print(len(response.candidates))
        if response.candidates[0].content.parts:
            generated_text = response.candidates[0].content.parts[0].text

    # Check if the response has 'parts' and it's not empty
    elif hasattr(response, "parts") and response.parts:
        # print(2)
        # Access the first part's text assuming it's the relevant content
        generated_text = response.parts[0].text

    elif hasattr(response, "text") and response.text:
        # print(3)
        # Access the text directly
        generated_text = response.text
    else:
        # Handle cases where no parts were generated
        # print("No content generated for the given input.")
        generated_text = "No content generated."

    # print("Content_generated: ", generated_text)
    return generated_text


response = generate_answer_1()
time.sleep(1)
print(response[:1000])


# *Assistant:* Hello! I'd be happy to assist you in your job search. With your interest in gaming, several job opportunities might be a good fit.

# *Here are some potential job roles you might consider:*

# * *Game Developer:* Design, develop, and test video games.
# * *Game Designer:* Create the concepts, characters, and storylines for video games.
# * *Game Tester:* Play and evaluate video games to identify and resolve bugs.
# * *Game Journalist:* Write articles, reviews, and news about the gaming industry.
# * *Community Manager:* Engage with gaming communities and manage online platforms.

# *You might also consider exploring the following industries:*

# * *Entertainment:* Game studios, production companies, and streaming platforms.
# * *Software Development:* Companies specializing in game development software.
# * *Social Media:* Platforms that focus on gaming content.

# *To enhance your job search:*

# * *Showcase your skills:* Highlight your experience in gaming, game design, or related areas.
# * *Network:* Attend industry events and connect with professionals in the gaming community.
# * *Build a portfolio:* Create a portfolio of your game projects or writing samples.
# * *Use job search engines:* Utilize platforms like Indeed, LinkedIn, and Glassdoor to find relevant job openings.
# * *Consider education or certifications:* Pursuing a degree or certification in game development or a related field can boost your credibility.

# Remember, finding the right job takes time and effort. Stay persistent, explore different options, and don't hesitate to reach out for additional support
