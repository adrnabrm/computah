# Computah

Local voice assistant. Wake word → listen → reply → remember.

## Stack

- **Wake word:** livekit-wakeword (`computah.onnx`)
- **STT:** faster-whisper (`tiny`)
- **LLM:** Ollama via smolagents / LiteLLM (default `qwen3.5:4b`)
- **TTS:** Piper (`en_US-lessac-medium`)
- **Memory:** short-term chat history across turns

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install and run [Ollama](https://ollama.com/), then pull the model:

```bash
ollama pull qwen3.5:4b
```

```bash
cp .env.example .env
```

```env
COMPUTAH_MODEL=qwen3.5:4b
OLLAMA_BASE=http://localhost:11434
WAKEWORD_MODEL_PATH=models/wakeword/computah.onnx
PIPER_VOICE_PATH=models/tts/en_US-lessac-medium.onnx
```

Load env before running (`export $(grep -v '^#' .env | xargs)` or similar).

## Run

```bash
python main.py
```

Ctrl+C to quit.

Loop: wake word → record → Whisper → Ollama (with chat memory) → Piper → repeat.

## Wake word training

Optional. Config: `scripts/wakeword/configs/prod.yaml`.

```bash
cd scripts/wakeword
python train.py
```

Exports ONNX under `models/wakeword/`. Point `WAKEWORD_MODEL_PATH` at it.
