from collections import deque

from smolagents import ChatMessage
from smolagents.models import MessageRole


class Memory:
    def __init__(self, max_size: int = 20):
        self.memory = deque(maxlen=max_size)

    def add(self, user: str, assistant: str) -> None:
        self.memory.append(
            ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": user}])
        )
        self.memory.append(
            ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": assistant}])
        )

    def get(self) -> list[ChatMessage]:
        return list(self.memory)
