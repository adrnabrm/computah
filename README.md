# Computah

Voice assistant. Wake word → listen → reply → remember. Can web search when needed.

## Stack

- **Wake word:** livekit-wakeword (`computah.onnx`)
- **STT:** faster-whisper (`tiny`)
- **LLM:** Gemini via smolagents / LiteLLM (`gemini-3.1-flash-lite`)
- **Tools:** DuckDuckGo web search
- **TTS:** Piper (`en_US-lessac-medium`)
- **Memory:** short-term chat history across turns

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

```env
GEMINI_API_KEY=your_key_here
WAKEWORD_MODEL_PATH=models/wakeword/computah.onnx
PIPER_VOICE_PATH=models/tts/en_US-lessac-medium.onnx
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey). Load env before running (`export $(grep -v '^#' .env | xargs)` or similar).

## Run

```bash
python main.py
```

Ctrl+C to quit.

Loop: wake word → record → Whisper → Gemini (memory + optional web search) → Piper → repeat.

## Wake word training

Optional. Config: `scripts/wakeword/configs/prod.yaml`.

```bash
cd scripts/wakeword
python train.py
```

Exports ONNX under `models/wakeword/`. Point `WAKEWORD_MODEL_PATH` at it.
