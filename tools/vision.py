import os
import subprocess
import tempfile

from PIL import Image
from smolagents import ChatMessage, LiteLLMModel
from smolagents.models import MessageRole

VISION_TOOL = {
    "type": "function",
    "function": {
        "name": "look_at_screen",
        "description": (
            "Capture the user's screen and answer a question about what is visible. "
            "Use when they ask what is on screen, to read UI text, or to describe what they are looking at."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "What to look for or answer from the screenshot",
                }
            },
            "required": ["question"],
        },
    },
}


class Vision:
    def __init__(
        self,
        api_key: str | None,
        model_id: str = "gemini/gemini-3.1-flash-lite",
        verbose: bool = False,
    ):
        self._verbose = verbose
        self.model = LiteLLMModel(
            model_id=model_id,
            api_key=api_key,
            max_tokens=512,
        )

    def look(self, question: str) -> str:
        path = self._screenshot()
        try:
            if self._verbose:
                print(f"[Vision] question={question!r} path={path!r}")
            image = Image.open(path)
            response = self.model.generate(
                [
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[
                            {"type": "text", "text": question},
                            {"type": "image", "image": image},
                        ],
                    )
                ]
            )
            return response.content or "Could not describe the screen."
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _screenshot(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        subprocess.run(["screencapture", "-x", path], check=True)
        return path
