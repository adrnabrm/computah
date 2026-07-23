"""
# TODO:
- add a way to remove all memories (not as tool to be executed but as a feature of the memory class)
"""

import uuid
from enum import Enum
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import GoogleGeminiEmbeddingFunction

class LongTermMemoryMessage(str, Enum):
    SAVED = "Saved."
    ALREADY_SAVED = "Already saved."
    FORGOTTEN = "Forgotten."
    UPDATED = "Updated."
    CANCELLED = "Cancelled. Nothing was changed."
    NO_MEMORIES = "No memories found."
    NOT_FOUND = "Memory not found. Nothing was changed."

REMEMBER_TOOL = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": "Save a durable fact about the user or their world for later. Use for names, preferences, people, places, and ongoing projects. Do not save temporary conversation details.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "One clear sentence about the user to store, written as 'The user ...'",
                }
            },
            "required": ["text"],
        },
    },
}

RECALL_TOOL = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": "Search long-term memories for facts that may answer the user. Use when conversation history is not enough.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short search query for the memory to find",
                }
            },
            "required": ["query"],
        },
    },
}

FORGET_TOOL = {
    "type": "function",
    "function": {
        "name": "forget",
        "description": "Delete one saved long-term memory that matches the query. Use when the user asks to forget or remove a fact.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What fact to delete, phrased like a stored memory or recall query (e.g. 'user's name', 'The user's dog is named Duke'). Not a single word unless that is the whole fact.",
                }
            },
            "required": ["query"],
        },
    },
}

UPDATE_TOOL = {
    "type": "function",
    "function": {
        "name": "update",
        "description": "Replace one saved long-term memory with a new fact. Use when the user corrects or changes something already remembered.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What fact to replace, phrased like a stored memory or recall query (e.g. 'user's name'). Not a single word unless that is the whole fact.",
                },
                "text": {
                    "type": "string",
                    "description": "The new fact to store, written as 'The user ...'",
                },
            },
            "required": ["query", "text"],
        },
    },
}


class LongTermMemory:
    def __init__(
        self,
        path: str,
        verbose: bool = False,
        confidence_threshold: float = 0.35,
        duplicate_threshold: float = 0.2,
    ):
        """ Initialize the long term memory. """
        self._verbose = verbose
        self.confidence_threshold = confidence_threshold
        self.duplicate_threshold = duplicate_threshold
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=path)
            self.collection = self.client.get_or_create_collection(
                name="long_term",
                embedding_function=GoogleGeminiEmbeddingFunction(),
            )
        except Exception as e:
            print(f"[LongTermMemory] Error initializing long term memory: {e}")
            raise e

    def remember(self, text: str) -> str:
        """Remember a piece of information, skipping near-duplicates."""
        if self.collection.count() > 0:
            result = self.collection.query(query_texts=[text], n_results=1)
            doc = result["documents"][0][0]
            distance = result["distances"][0][0]
            if self._verbose:
                print(f"[LongTermMemory] dedupe text={text!r} doc={doc!r} distance={distance}")
            if doc.strip().lower() == text.strip().lower() or distance < self.duplicate_threshold:
                return LongTermMemoryMessage.ALREADY_SAVED.value

        self.collection.add(documents=[text], ids=[str(uuid.uuid4())])
        if self._verbose:
            print(f"[LongTermMemory] remember text={text!r}")
        return LongTermMemoryMessage.SAVED.value

    def recall(self, query: str) -> str:
        """ Recall information from persistent memory based on a query. """
        n = min(3, self.collection.count())
        if n == 0:
            return LongTermMemoryMessage.NO_MEMORIES.value
        
        # Query the collection
        result = self.collection.query(query_texts=[query], n_results=n)
        # Retrieve the memory pieces and distances (how far off the query is from the memory piece)
        docs = result["documents"][0]
        distances = result["distances"][0]

        if self._verbose:
            for doc, distance in zip(docs, distances):
                print(f"[LongTermMemory] recall query={query!r} doc={doc!r} distance={distance}")

        # Filter out the documents that are too far off the query
        kept = [
            doc
            for doc, distance in zip(docs, distances)
            # Lower distance means closer match
            if distance < self.confidence_threshold
        ]
        if not kept:
            return LongTermMemoryMessage.NO_MEMORIES.value
        return "\n".join(kept)

    def find_closest(self, query: str) -> tuple[str, str] | None:
        """Return (id, doc) for the closest memory under the threshold, else None."""
        if self.collection.count() == 0:
            return None

        result = self.collection.query(query_texts=[query], n_results=1)
        doc = result["documents"][0][0]
        distance = result["distances"][0][0]
        memory_id = result["ids"][0][0]

        if self._verbose:
            print(f"[LongTermMemory] find_closest query={query!r} doc={doc!r} distance={distance}")

        if distance >= self.confidence_threshold:
            return None
        return memory_id, doc

    def delete(self, memory_id: str) -> None:
        self.collection.delete(ids=[memory_id])
        if self._verbose:
            print(f"[LongTermMemory] deleted id={memory_id!r}")

    def forget(self, query: str) -> str:
        """Delete the closest matching memory, or refuse if nothing is close enough."""
        match = self.find_closest(query)
        if not match:
            return LongTermMemoryMessage.NOT_FOUND.value
        memory_id, _ = match
        self.delete(memory_id)
        return LongTermMemoryMessage.FORGOTTEN.value

    def update(self, query: str, text: str) -> str:
        """Forget the closest match, then remember the new text."""
        result = self.forget(query)
        if result != LongTermMemoryMessage.FORGOTTEN.value:
            return result
        self.remember(text)
        return LongTermMemoryMessage.UPDATED.value
