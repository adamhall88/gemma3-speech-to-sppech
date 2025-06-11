# Gemma3 Speech-to-Speech

This repository contains a simple prototype for a local AI voice tutor using [Whisper](https://github.com/openai/whisper), [Ollama](https://ollama.com/), and [Coqui TTS](https://github.com/coqui-ai/TTS).

## Setup
1. Ensure the following Python packages are installed: `openai-whisper`, `sounddevice`, `numpy`, `requests`, `playsound`, `TTS`.
2. Start Gemma 4B locally in another terminal:
   ```bash
   ollama run gemma:4b
   ```
   This launches an API server at `http://localhost:11434`.
3. Open a terminal and **change into this repository's directory** before running the script.

## Running
From the project directory, run `voice_tutor.py` to start the interactive loop:

```bash
python3 voice_tutor.py
```

The script records a short audio clip, transcribes it with Whisper, sends the text to Gemma via Ollama, then speaks the response using Coqui TTS. Press `Ctrl+C` to stop.
