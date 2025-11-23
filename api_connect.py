import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
from google import genai

load_dotenv()

def elevenlabs():
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    client = ElevenLabs(api_key=api_key)
    # client = ElevenLabs(api_key="ELEVENLABS_API_KEY")

    text_to_speak = "Hello! This is a voice generated using the ElevenLabs Python SDK."

    # Get raw response with headers
    response = client.text_to_speech.convert(
        text=text_to_speak,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_flash_v2", # 50% cheaper
        # model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    print(f"Generating and playing: '{text_to_speak}'")
    play(response)

def gemini():
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Explain how AI works in a few words"
    )
    print(response.text)

# gemini()
elevenlabs()