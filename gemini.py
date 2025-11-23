import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

def promptGemini(prompt="Explain how AI works in a few words"):
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text