import os
import time
import sounddevice as sd
import numpy as np
import whisper
import requests
from playsound import playsound
from TTS.api import TTS

# Configuration
DURATION = 5  # seconds to record
SAMPLE_RATE = 16000
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:4b"
WHISPER_MODEL = "base"

tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)


def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    """Record audio from the microphone."""
    try:
        print("Listening...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        return audio.flatten()
    except Exception as e:
        print(f"Recording error: {e}")
        return None


def transcribe_audio(audio):
    """Transcribe audio using Whisper."""
    try:
        model = whisper.load_model(WHISPER_MODEL)
        result = model.transcribe(audio, fp16=False)
        text = result.get("text", "").strip()
        print(f"You: {text}")
        return text
    except Exception as e:
        print(f"Whisper error: {e}")
        return None


def ask_gemma(prompt):
    """Send prompt to Gemma via Ollama API."""
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "").strip()
        print(f"Gemma: {answer}")
        return answer
    except Exception as e:
        print(f"Ollama API error: {e}")
        return None


def speak_text(text):
    """Speak text aloud using Coqui TTS."""
    try:
        wav_path = "gemma_reply.wav"
        tts.tts_to_file(text=text, file_path=wav_path)
        playsound(wav_path)
        os.remove(wav_path)
    except Exception as e:
        print(f"TTS error: {e}")


def main():
    print("Starting voice tutor. Press Ctrl+C to stop.")
    while True:
        audio = record_audio()
        if audio is None:
            continue
        text = transcribe_audio(audio)
        if not text:
            continue
        reply = ask_gemma(text)
        if not reply:
            continue
        speak_text(reply)
        time.sleep(0.5)  # brief pause between iterations


if __name__ == "__main__":
    main()

