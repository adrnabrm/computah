import json
import os

from smolagents import ChatMessage, LiteLLMModel
from smolagents.models import MessageRole

from tools.web_search import WEB_SEARCH_TOOL, WebSearchStatus, web_search
from utils.audio_handler import AudioHandler
from utils.memory import Memory

MODEL_ID = os.getenv("COMPUTAH_MODEL", "qwen3.5:4b")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SYSTEM_PROMPT = """
You are a voice assistant. Answer the user's latest question using conversation history when it is enough.

Tools:
- Use web_search only for live or external facts you cannot know from memory (weather, news, scores, current events).
- Do not use tools to recall what the user just said or what you already answered.
- Do not use tools for general knowledge or everyday how-tos unless the user asks for something current from the web.
- Answer only the latest user question. Do not mix in older topics unless they ask about them.

Response style (unbreakable):
- Conversational, 2-4 sentences.
- Plain text only: letters, numbers, and punctuation.
- No markdown, code blocks, bullet points, tables, or URLs.
"""

class Computah:

    def __init__(self):
        print("Initializing Computah...")
        # Initialize the model
        try:
            self.model = LiteLLMModel(
                model_id="gemini/gemini-3.1-flash-lite",
                api_key=GEMINI_API_KEY,
                num_ctx=8192,
                max_tokens=256,
            )
            # Initialize the tools
            self.tools = [WEB_SEARCH_TOOL]
            self.tool_fns = {"web_search": web_search}
            self.max_tool_rounds = 3

            # Initialize the memory
            self.memory = Memory()
            # Initialize the audio handler
            self.audio_handler = AudioHandler()
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise e
        print("Computah initialized!")

    def run(self) -> None:
        print("Starting Computah...")
        while True:
            try:
                self._listen_for_wakeword()

                user_query_transcript = self._capture_user_audio()
                if user_query_transcript:
                    print(f"User said: {user_query_transcript}")
                    response = self._query_model(user_query_transcript)
                else:
                    raise Exception("No user query transcript captured!")

                self._speak(response)
                self.memory.add(user_query_transcript, response)
            except KeyboardInterrupt:
                print("Computah shutting down...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

    def _speak(self, input: str) -> None:
        """Speak the response to the user."""
        self.audio_handler.speak(input)

    def _listen_for_wakeword(self) -> bool:
        """Listen for the wakeword and return True if detected, False otherwise."""
        if self.audio_handler.listen_for_wakeword():
            print("Wakeword detected!")
            return True
        raise Exception("No wakeword detected!")
    
    def _capture_user_audio(self) -> str:
        """Capture audio from the user and transcribe it."""
        print("Capturing user audio...")
        return self.audio_handler.capture_audio()

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
                print(f"Tool call: {name} with arguments: {args}")
                self._speak(f"Using tool {name}")

                if isinstance(args, str):
                    args = json.loads(args) if args else {}
                try:
                    result = self.tool_fns[name](**args)
                    success = result != WebSearchStatus.NO_RESULTS
                except Exception as e:
                    print(f"Error using tool {name}: {e}")
                    result = f"Error using tool {name}: {e}"

                messages.append(
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[{"type": "text", "text": f"Tool result ({name}):\n{result}"}],
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
