from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
import os

client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))

def playtextToSpeech(text):
    response = client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_flash_v2", # 50% cheaper
        # model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    print(f"Generating and playing: '{text}'")
    play(response)

def playSoundEffect(description):
    audio = client.text_to_sound_effects.convert(text=description)
    print(f"Generating sound effect: '{description}'")
    play(audio)

def playMusic(description, duration_ms=10000):
    track = client.music.compose(
        prompt=description,
        music_length_ms=duration_ms,
    )

    play(audio)

def test():
    text_to_speak = "Hello! This is a voice generated using the ElevenLabs Python SDK."

    # Get raw response with headers
    response = client.text_to_speech.convert(
        text=text_to_speak,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_flash_v2", # 50% cheaper
        # model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    print(f"Generating Voice: '{text_to_speak}'")
    play(response)