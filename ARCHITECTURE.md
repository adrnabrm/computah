# Computah Architecture

Mental model of the system for humans and coding agents. Read this before changing code.

## What it is

Local voice assistant. Single process, no HTTP API.

```
wake word → record → Whisper STT → Gemini (+ tools) → Piper TTS → remember turn → wait again
```

Entry: `python main.py` → `Computah().run()`.

## High-level design

Everything hangs off one orchestrator: `Computah` in `agent.py`.

```
main.py
   └── Computah (agent.py)
         ├── AudioHandler     wake / record / STT / TTS
         ├── Memory           short-term chat (in-process deque)
         ├── LiteLLMModel     Gemini chat (tool loop owned by Computah)
         └── tools
               ├── WebSearch.search
               ├── Vision.look
               └── LongTermMemory  (+ Computah._forget / _update wrappers)
```

No smolagents `CodeAgent` / `ToolCallingAgent`. smolagents is used only for `LiteLLMModel`, `ChatMessage`, and `MessageRole`. The tool loop is hand-rolled in `_query_model`.

## Directory structure

```
computah/
├── main.py                 # entry
├── agent.py                # Computah orchestrator, system prompt, tool loop
├── requirements.txt
├── .env.example
├── tools/
│   ├── web_search.py       # DuckDuckGo + page fetch
│   ├── vision.py           # macOS screenshot + Gemini vision
│   └── longterm_mem.py     # Chroma remember/recall/forget/update
├── utils/
│   ├── audio_handler.py    # wake word, record, Whisper, Piper
│   └── memory.py           # short-term deque
├── models/
│   ├── wakeword/           # computah.onnx (+ train eval artifacts)
│   └── tts/                # Piper en_US-lessac-medium.onnx
├── scripts/wakeword/       # optional wake-word training pipeline
│   ├── train.py
│   └── configs/prod.yaml
└── data/chroma/            # persistent LTM (gitignored)
```

Runtime side effects in cwd: `user_audio.wav`, `output.wav`.

## Core loop

`Computah.run()`:

```
while True:
  listen_for_wakeword()          # blocks; Ctrl+C exits
  transcript = capture_audio()   # record → Whisper
  response = _query_model(transcript)
  speak(response)
  memory.add(transcript, response)
```

Non-`KeyboardInterrupt` errors print and continue (back to wake listen). Empty transcript raises and is swallowed by that catch.

## Data flow (one turn)

```
mic
 │
 ├─ WakeWordListener (threshold 0.1) ──► beep 880 Hz
 │
 ├─ record (16 kHz, silence stop) ──► user_audio.wav ──► beep 660 Hz
 │                                         │
 │                                    Whisper tiny
 │                                         │
 ▼                                         ▼
messages = [SYSTEM_PROMPT, *short_term, USER(transcript)]
                │
                ▼
     Gemini.generate(tools, tool_choice=auto)   ← up to 3 rounds
                │
        ┌───────┴────────┐
        │ no tool_calls  │ has tool_calls
        ▼                ▼
   return text     speak "Using tool {name}"
                   run tool_fns[name](**args)
                   append USER: "[Label]\n{result}"
                   loop (or final generate without tools)
                │
                ▼
         Piper → output.wav → play
                │
                ▼
         Memory.add(user, assistant)
```

### Message construction

Every model call gets:

1. `SYSTEM` — `SYSTEM_PROMPT` (tool policy, evidence rules, voice style)
2. Short-term history — up to 20 messages (10 user/assistant turns)
3. Current `USER` — this turn’s transcript
4. (During tool rounds) more `USER` blobs labeled `[WebSearch]`, `[Vision]`, or `[LongTermMemory]`

Tool results are **not** native tool-role messages. They are labeled user text. Labels live in `TOOL_RESULT_LABELS` and must stay aligned with the system prompt’s evidence rules.

### Tool loop rules

- Max 3 rounds with tools.
- Multiple tool calls in one round all run before the next generate.
- After the cap: one final `generate(messages)` with no tools.

## Tools

Schema + handler are separate. To add a tool:

1. OpenAI-style function dict (`*_TOOL`) in `tools/`
2. Handler method
3. Register in `Computah.tools`, `tool_fns`, and `TOOL_RESULT_LABELS`

| Name | Handler | Role |
|------|---------|------|
| `web_search` | `WebSearch.search` | DDGS top 3; fetch body of top 2 (≤2500 chars each) |
| `look_at_screen` | `Vision.look` | `screencapture` → Gemini multimodal (macOS only) |
| `remember` | `LongTermMemory.remember` | Persist a `"The user ..."` fact |
| `recall` | `LongTermMemory.recall` | Similarity search over saved facts |
| `forget` | `Computah._forget` | Find → voice confirm → delete |
| `update` | `Computah._update` | Find → voice confirm → replace |

`forget` / `update` go through `_confirm_yes`: speak a prompt, capture audio (no wake word), accept yes-ish answers. Cancelled/not-found return status strings the prompt tells the model to honor.

### Long-term memory protocol

Chroma collection `"long_term"` with `GoogleGeminiEmbeddingFunction`. Distances: lower = closer.

| Threshold | Default | Meaning |
|-----------|---------|---------|
| `confidence_threshold` | `0.45` | Keep/match if distance below this |
| `duplicate_threshold` | `0.1` | Skip `remember` if too similar |

Status prefixes the model must quote after write tools (see `LongTermMemoryMessage`):

- `Saved:` / `Already saved:` / `Updated to:` / `Forgotten:`
- `Cancelled. Nothing was changed.` / `Memory not found. Nothing was changed.` / `No memories found.`

## Memory model (two tiers)

| Tier | Class | Store | Lifetime |
|------|-------|-------|----------|
| Short-term | `utils.memory.Memory` | `deque(maxlen=20)` | process |
| Long-term | `tools.longterm_mem.LongTermMemory` | Chroma at `LONG_TERM_MEMORY_PATH` | disk |

Short-term is always updated after a spoken reply. Long-term only changes when the model calls remember/update/forget.

Intended tool order (enforced by prompt, not code): history → `recall` → `web_search`.

## Audio stack

`AudioHandler` owns the mic/speaker path.

| Stage | Implementation |
|-------|----------------|
| Wake word | `livekit-wakeword` `WakeWordModel` + `WakeWordListener`, threshold `0.1` |
| Record | sounddevice, 16 kHz mono int16, start when RMS > 500, stop after 1 s silence or 30 s |
| STT | faster-whisper `tiny`, CPU int8 |
| TTS | Piper voice from `PIPER_VOICE_PATH` |

SIGINT during wake listen uses a threading `Event` so Ctrl+C works with PyAudio.

Vision uses macOS `screencapture -x`. It will fail on other OSes.

## Config

Env vars are read via `os.getenv`. `python-dotenv` is in requirements but **not** called — export `.env` yourself before run.

| Variable | Used? | Notes |
|----------|-------|-------|
| `GEMINI_API_KEY` | yes | Chat, vision, Chroma Gemini embeddings |
| `WAKEWORD_MODEL_PATH` | yes | Required or `AudioHandler` raises |
| `PIPER_VOICE_PATH` | yes | Required or `AudioHandler` raises |
| `LONG_TERM_MEMORY_PATH` | yes | Default `data/chroma` |
| `COMPUTAH_MODEL` | no | Read but unused; model hardcoded |
| `OLLAMA_BASE` | no | Read but unused |

Hardcoded chat model: `gemini/gemini-3.1-flash-lite` (`num_ctx=8192`, `max_tokens=256`). Vision uses the same model id with `max_tokens=512`.

## External integrations

| System | Package / API | Purpose |
|--------|---------------|---------|
| Gemini | litellm via smolagents | Chat + vision |
| Gemini embeddings | chromadb `GoogleGeminiEmbeddingFunction` | LTM vectors |
| DuckDuckGo | `ddgs` | Search |
| httpx | page fetch for search | |
| Whisper | faster-whisper | STT |
| livekit-wakeword | detection (+ training scripts) | |
| Piper | piper-tts | TTS |
| Chroma | chromadb PersistentClient | LTM |
| sounddevice / wavio / PyAudio | mic, speakers, wakeword I/O | |

`fastapi` / `uvicorn` appear in requirements but are unused by app code.

## Conventions agents must respect

1. **Orchestration stays in `agent.py`.** Tools stay thin; voice confirmation for destructive LTM stays on `Computah`.
2. **Add tools via schema + `tool_fns` + label.** Do not introduce smolagents agent runners unless intentionally rewriting the loop.
3. **Keep LTM status prefixes stable.** System prompt and reply style depend on them.
4. **Tool results remain labeled USER text** unless you change both the loop and the prompt.
5. **No new packages** unless asked; match existing patterns.
6. **Run from repo root** so `from tools...` / `from utils...` resolve (no package `__init__.py`s).
7. **Vision is macOS-specific.**
8. **Optional wake-word training** lives under `scripts/wakeword/`; production model is `models/wakeword/computah.onnx`.

## Out of scope today

- HTTP / multi-user API
- Auto-loading `.env`
- Wiring `COMPUTAH_MODEL` / Ollama
- Clear-all LTM (TODO in `longterm_mem.py`)
- Packaging as an installable library
