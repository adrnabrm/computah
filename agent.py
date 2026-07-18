import os
from smolagents import ChatMessage, LiteLLMModel
from smolagents.models import MessageRole
from utils.audio_handler import AudioHandler
from utils.memory import Memory

MODEL_ID = os.getenv("COMPUTAH_MODEL", "qwen3.5:4b")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
SYSTEM_PROMPT = """
You are a voice assistant tasked with answering questions and/or performing tasks whether or not tools are available to you.
You will receive a transcript of the user's query and you will need to respond following these strict rules that are unbreakable:
- Respond in a conversational style.
- Limit your response to 2-4 sentences.
- Do not use any other characters other than plain text alphabet letters, numbers, and punctuation. 
- No markdown, no code blocks, no bullet points, no tables, no URLS, or any formatting that is other than plain text conversation.
"""

class Computah:

    def __init__(self):
        print("Initializing Computah...")
        # Initialize the model
        self.model = LiteLLMModel(
            model_id=f"ollama_chat/{MODEL_ID}",
            api_base=OLLAMA_BASE,
            num_ctx=8192,
            max_tokens=256,
        )
        # Initialize the memory
        self.memory = Memory()
        # Initialize the audio handler
        self.audio_handler = AudioHandler()
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
        response = self.model.generate(messages)
        return response.content
