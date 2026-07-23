"""
# TODO:
- add duplicate detection when remembering, eg if user says "I have a cat named Fluffy" and then "I have a cat named Fluffy", don't remember the second one
- add a way to remove all memories (not as tool to be executed but as a feature of the memory class)
"""

import uuid
from enum import Enum
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import GoogleGeminiEmbeddingFunction

class LongTermMemoryMessage(str, Enum):
    SAVED = "Saved."
    FORGOTTEN = "Forgotten."
    CANCELLED = "Cancelled. Nothing was forgotten."
    NO_MEMORIES = "No memories found."
    NOT_FOUND = "Memory not found. Nothing was forgotten."

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


class LongTermMemory:
    def __init__(self, path: str, verbose: bool = False, confidence_threshold: float = 0.6):
        """ Initialize the long term memory. """
        self._verbose = verbose
        self.confidence_threshold = confidence_threshold
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=path)
            self.collection = self.client.get_or_create_collection(
                name="long_term",
                embedding_function=GoogleGeminiEmbeddingFunction(),
            )
        except Exception as e:
            print(f"Error initializing long term memory: {e}")
            raise e

    def remember(self, text: str) -> str:
        """ Remember a piece of information from persistent memory."""
        self.collection.add(documents=[text], ids=[str(uuid.uuid4())])
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
                print(f"recall query={query!r} doc={doc!r} distance={distance}")

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
            print(f"find_closest query={query!r} doc={doc!r} distance={distance}")

        if distance >= self.confidence_threshold:
            return None
        return memory_id, doc

    def delete(self, memory_id: str) -> None:
        self.collection.delete(ids=[memory_id])
