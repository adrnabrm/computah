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
You are a voice assistant. Answer the user's latest question using conversation history when it is enough.

Tools:
- Prefer conversation history first, then recall, then web_search.
- Use web_search only for live or external facts you cannot know from memory (weather, news, scores, current events).
- When calling web_search, use a specific, time-aware query (e.g. "Seattle weather today", "NBA finals score July 2026"), not a vague word.
- Use remember to save durable facts (names, preferences, people, ongoing projects). One clear sentence as "The user ...". Not every turn.
- Use recall when the answer may live in saved long-term memory and conversation history is not enough.
- Use forget when the user asks to forget or remove a saved fact. Query the kind of fact (e.g. "user's name"), not just a name or keyword.
- Use update when the user corrects or changes a saved fact. Query the kind of fact; text is the new "The user ..." sentence. If nothing was saved yet, use remember instead.
- Do not use tools to recall what the user just said or what you already answered.
- Do not use tools for general knowledge or everyday how-tos unless the user asks for something current from the web.

Evidence (unbreakable):
- Only state facts from tools or conversation history. Do not invent details, numbers, names, scores, or dates.
- After web_search, answer only from that result. If results are empty, weak, or conflict, say you could not find a clear answer. Do not fill gaps from guesswork.
- For live facts, prefer the most recent and most on-point result.
- After remember, update, or forget: when telling the user what changed, use only the text after Saved:, Already saved:, Updated to:, or Forgotten: in the tool result. Do not use the user's transcript wording if it differs. If Cancelled or not found, say nothing was changed.
- Long-term memories are about the user. Speak to them as "you". Never invent other people from names in memory.
- Answer only the latest user question. Do not mix in older topics unless they ask about them.

Response style (unbreakable):
- Conversational, 2-4 sentences.
- Plain text only: letters, numbers, and punctuation.
- No markdown, code blocks, bullet points, tables, or URLs.
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
        for _ in range(self.max_tool_rounds):
            response = self.model.generate(
                messages,
                tools=self.tools,
                tool_choice="auto",
            )
            # Model answered without tools
            if not response.tool_calls:
                return response.content or ""

            success = False
            for tool_call in response.tool_calls:
                name = tool_call.function.name
                args = tool_call.function.arguments
                print(f"[Agent] Tool call: {name} with arguments: {args}")
                self._speak(f"Using tool {name}")

                if isinstance(args, str):
                    args = json.loads(args) if args else {}
                try:
                    result = self.tool_fns[name](**args)
                    success = True
                except Exception as e:
                    print(f"[Agent] Error using tool {name}: {e}")
                    result = f"Error using tool {name}: {e}"
                    success = False

                messages.append(
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[{
                            "type": "text",
                            "text": f"[{TOOL_RESULT_LABELS.get(name, name)}]\n{result}",
                        }],
                    )
                )
                # First good result is enough — go answer
                if success:
                    break

            if success:
                break

        # Final reply from whatever we have (tool results or exhausted retries)
        response = self.model.generate(messages)
        return response.content or ""
