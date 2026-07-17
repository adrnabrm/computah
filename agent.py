import os
from smolagents import ChatMessage, LiteLLMModel
from smolagents.models import MessageRole
from utils import AudioHandler

MODEL_ID = os.getenv("DUKEBOT_MODEL", "qwen3.5:4b")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")

class Computah:

    def __init__(self):
        print("Initializing Computah...")
        # Initialize the model
        self.model = LiteLLMModel(
            model_id=f"ollama_chat/{MODEL_ID}",
            api_base=OLLAMA_BASE,
            num_ctx=8192,
        )
        # Initialize the audio handler
        self.audio_handler = AudioHandler()
        print("Computah initialized!")

    def listen(self) -> str:
        print("Listening for audio...")
        transcript = self.audio_handler.capture_audio()
        print("Transcript captured!")
        return transcript

    def run(self, input: str) -> str:
        response = self.model.generate([
            ChatMessage(
                role=MessageRole.USER,
                content=[{"type": "text", "text": input}],
            ),
        ])
        return response.content