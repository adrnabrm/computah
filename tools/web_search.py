from enum import Enum
from html import unescape
import re

import httpx
from ddgs import DDGS

class WebSearchStatus(str, Enum):
    NO_RESULTS = "No results found."


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information (news, weather, scores, live facts). "
            "Use a specific, time-aware query. Returns titles, urls, snippets, and fetched page text. "
            "Not for conversation history or general knowledge."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Specific search query, preferably time-aware",
                }
            },
            "required": ["query"],
        },
    },
}

_FETCH_COUNT = 2
_MAX_CHARS = 2500
_TIMEOUT = 5.0
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<script[^>]*>.*?</script>", re.I | re.S)
_STYLE_RE = re.compile(r"<style[^>]*>.*?</style>", re.I | re.S)
_WS_RE = re.compile(r"\s+")


class WebSearch:
    def __init__(self, verbose: bool = False):
        self._verbose = verbose

    def search(self, query: str) -> str:
        """Search the web and auto-fetch top page content."""
        results = DDGS().text(query, max_results=3)
        if self._verbose:
            print(f"[WebSearch] query={query!r} results={len(results) if results else 0}")
        if not results:
            return WebSearchStatus.NO_RESULTS.value

        blocks = []
        for i, r in enumerate(results, 1):
            title = r.get("title") or ""
            url = r.get("href") or r.get("link") or ""
            snippet = r.get("body") or ""
            content = ""
            if i <= _FETCH_COUNT and url:
                content = self._fetch_page(url)
                if self._verbose:
                    print(f"[WebSearch] fetch url={url!r} chars={len(content)}")
            elif self._verbose:
                print(f"[WebSearch] {title}: {snippet}")

            parts = [f"{i}. title: {title}", f"   url: {url}", f"   snippet: {snippet}"]
            if content:
                parts.append(f"   content: {content}")
            blocks.append("\n".join(parts))

        return "\n".join(blocks)

    def _fetch_page(self, url: str) -> str:
        try:
            with httpx.Client(follow_redirects=True, timeout=_TIMEOUT) as client:
                resp = client.get(
                    url,
                    headers={"User-Agent": "Computah/1.0"},
                )
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "")
                if "html" not in ctype and "text" not in ctype:
                    return ""
                return self._html_to_text(resp.text)[:_MAX_CHARS]
        except Exception as e:
            if self._verbose:
                print(f"[WebSearch] fetch failed url={url!r} err={e}")
            return ""

    def _html_to_text(self, html: str) -> str:
        text = _SCRIPT_RE.sub(" ", html)
        text = _STYLE_RE.sub(" ", text)
        text = _TAG_RE.sub(" ", text)
        text = unescape(text)
        return _WS_RE.sub(" ", text).strip()
