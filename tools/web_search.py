from enum import Enum
from ddgs import DDGS

class WebSearchStatus(str, Enum):
    NO_RESULTS = "No results found."


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information. Such as current news, weather, and other live and modern information. This is not intended for conversational history or general knowledge.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                }
            },
            "required": ["query"],
        },
    },
}


def web_search(query: str) -> str:
    """
    Search the web for current information.
    Returns a list of results that contain the title of the page and its body content.
    """
    results = DDGS().text(query, max_results=5)
    if not results:
        return WebSearchStatus.NO_RESULTS
    return "\n".join(f"{r['title']}: {r['body']}" for r in results)
