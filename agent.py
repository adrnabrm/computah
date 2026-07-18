import os
from smolagents import ChatMessage, LiteLLMModel
from smolagents.models import MessageRole
from utils.audio_handler import AudioHandler

MODEL_ID = os.getenv("COMPUTAH_MODEL", "qwen3.5:4b")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")

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
        # Initialize the audio handler
        self.audio_handler = AudioHandler()
        print("Computah initialized!")

    def run(self) -> None:
        print("Starting Computah...")
        self._listen_for_wakeword()
        user_query_transcript = self._capture_user_audio()
        if user_query_transcript:
            print(f"User said: {user_query_transcript}")
            return self._query_model(user_query_transcript)
        else:
            raise Exception("No user query transcript captured!")

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
        response = self.model.generate([
            ChatMessage(
                role=MessageRole.USER,
                content=[{"type": "text", "text": input}],
            ),
        ])
        return response.content