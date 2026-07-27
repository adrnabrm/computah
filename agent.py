import json
import os

from smolagents import ChatMessage, LiteLLMModel
from smolagents.models import MessageRole

from tools.longterm_mem import (
    FORGET_TOOL,
    RECALL_TOOL,
    REMEMBER_TOOL,
    UPDATE_TOOL,
    LongTermMemory,
    LongTermMemoryMessage,
)
from tools.web_search import WEB_SEARCH_TOOL, WebSearch
from utils.audio_handler import AudioHandler
from utils.memory import Memory

MODEL_ID = os.getenv("COMPUTAH_MODEL", "qwen3.5:4b")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LONG_TERM_MEMORY_PATH = os.getenv("LONG_TERM_MEMORY_PATH", "data/chroma")
TOOL_RESULT_LABELS = {
    "web_search": "WebSearch",
    "remember": "LongTermMemory",
    "recall": "LongTermMemory",
    "forget": "LongTermMemory",
    "update": "LongTermMemory",
}
SYSTEM_PROMPT = """
You are a voice assistant. Answer only the latest user question. Use conversation history when it is enough.

Tools (order: history → recall → web_search):
- recall: saved personal facts (name, location, prefs) when history lacks them. Query the kind of fact (e.g. "user's location").
- web_search: live facts only (weather, news, scores). Specific time-aware queries. Never "my location", "near me", or similar.
- If the user means "here" / no city named: recall location first, then web_search with that city.
- remember: durable "The user ..." facts. Not every turn.
- forget / update: query the kind of fact; update text is the new "The user ..." sentence. No match on update → remember instead.
- Do not tool-call for what was just said, or for general knowledge unless they want something current from the web.

Evidence (unbreakable):
- Only state facts from tools or history. No inventing details, numbers, names, scores, or dates.
- After web_search: answer only from that result. Empty/weak/conflict → say you could not find a clear answer.
- After remember/update/forget: quote only the text after Saved:, Already saved:, Updated to:, or Forgotten:. Not the transcript. Cancelled/not found → nothing changed.
- Long-term memories are about the user ("you"). Never invent other people from names in memory.

Response style (unbreakable):
- Conversational, 2-4 sentences. Plain text only. No markdown, code, bullets, tables, or URLs.
"""

class Computah:

    def __init__(self):
        print("[Agent] Initializing Computah...")
        # Initialize the model
        try:
            self.model = LiteLLMModel(
                model_id="gemini/gemini-3.1-flash-lite",
                api_key=GEMINI_API_KEY,
                num_ctx=8192,
                max_tokens=256,
            )
            # Initialize the memory
            self.memory = Memory()
            self.long_term = LongTermMemory(path=LONG_TERM_MEMORY_PATH, verbose=True)
            self.web_search = WebSearch(verbose=True)

            # Initialize the tools
            self.tools = [WEB_SEARCH_TOOL, REMEMBER_TOOL, RECALL_TOOL, FORGET_TOOL, UPDATE_TOOL]
            self.tool_fns = {
                "web_search": self.web_search.search,
                "remember": self.long_term.remember,
                "recall": self.long_term.recall,
                "forget": self._forget,
                "update": self._update,
            }
            self.max_tool_rounds = 3

            # Initialize the audio handler
            self.audio_handler = AudioHandler()
        except Exception as e:
            print(f"[Agent] Error initializing: {e}")
            raise e
        print("[Agent] Computah initialized!")

    def run(self) -> None:
        print("[Agent] Starting Computah...")
        while True:
            try:
                self._listen_for_wakeword()

                user_query_transcript = self._capture_user_audio()
                if user_query_transcript:
                    print(f"[Agent] User said: {user_query_transcript}")
                    response = self._query_model(user_query_transcript)
                else:
                    raise Exception("No user query transcript captured!")

                self._speak(response)
                self.memory.add(user_query_transcript, response)
            except KeyboardInterrupt:
                print("[Agent] Computah shutting down...")
                break
            except Exception as e:
                print(f"[Agent] Error: {e}")
                continue

    # Audio handling
    def _speak(self, input: str) -> None:
        """Speak the response to the user."""
        self.audio_handler.speak(input)

    def _listen_for_wakeword(self) -> bool:
        """Listen for the wakeword and return True if detected, False otherwise."""
        if self.audio_handler.listen_for_wakeword():
            print("[Agent] Wakeword detected!")
            return True
        raise Exception("No wakeword detected!")
    
    def _capture_user_audio(self) -> str:
        """Capture audio from the user and transcribe it."""
        print("[Agent] Capturing user audio...")
        return self.audio_handler.capture_audio()

    # Long-term memory handling
    def _confirm_yes(self, prompt: str) -> bool:
        self._speak(prompt)
        answer = (self._capture_user_audio() or "").strip().lower().strip(".,!?")
        print(f"[LongTermMemory] confirm answer={answer!r}")
        return answer.startswith("yes") or answer in ("yeah", "yep", "yup", "sure", "ok", "okay")

    def _forget(self, query: str) -> str:
        """Find a memory, confirm by voice, then delete if the user says yes."""
        match = self.long_term.find_closest(query)
        if not match:
            return LongTermMemoryMessage.NOT_FOUND.value

        _, doc = match
        if not self._confirm_yes(f"Are you sure you want to delete this memory: {doc}"):
            return LongTermMemoryMessage.CANCELLED.value

        return self.long_term.forget(query)

    def _update(self, query: str, text: str) -> str:
        """Find a memory, confirm by voice, then replace it if the user says yes."""
        match = self.long_term.find_closest(query)
        if not match:
            return LongTermMemoryMessage.NOT_FOUND.value

        _, doc = match
        if not self._confirm_yes(f"Replace this memory: {doc} with: {text}"):
            return LongTermMemoryMessage.CANCELLED.value

        return self.long_term.update(query, text)

    # Model handling
    def _query_model(self, input: str) -> str:
        """Query the model with the user input and return the response."""
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[{"type": "text", "text": SYSTEM_PROMPT}],
            ),
            *self.memory.get(),
            ChatMessage(
                role=MessageRole.USER,
                content=[{"type": "text", "text": input}],
            ),
        ]
        # Keep calling tools until the model answers, or we hit max_tool_rounds.
        for _ in range(self.max_tool_rounds):
            response = self.model.generate(
                messages,
                tools=self.tools,
                tool_choice="auto",
            )
            if not response.tool_calls:
                return response.content or ""

            # Run every tool in this round, append results, then loop so the
            # model can call another tool (e.g. recall → web_search).
            for tool_call in response.tool_calls:
                name = tool_call.function.name
                args = tool_call.function.arguments
                print(f"[Agent] Tool call: {name} with arguments: {args}")
                self._speak(f"Using tool {name}")

                if isinstance(args, str):
                    args = json.loads(args) if args else {}
                try:
                    result = self.tool_fns[name](**args)
                except Exception as e:
                    print(f"[Agent] Error using tool {name}: {e}")
                    result = f"Error using tool {name}: {e}"

                messages.append(
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[{
                            "type": "text",
                            "text": f"[{TOOL_RESULT_LABELS.get(name, name)}]\n{result}",
                        }],
                    )
                )

        # Cap hit — force a final answer with whatever tool results we have.
        response = self.model.generate(messages)
        return response.content or ""
