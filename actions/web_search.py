# web_search.py
import json
import sys
from pathlib import Path

from core.model_config import HELPER_MODEL


def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = _get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"


def _get_api_key() -> str:
    with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["gemini_api_key"]


def _response_text(response) -> str:
    """Extract only ordinary text parts, avoiding SDK thought-part warnings."""
    chunks: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
    return "".join(chunks).strip()


def _ddg_text(query: str, max_results: int = 8) -> list[dict]:
    from ddgs import DDGS

    results = []
    with DDGS(timeout=8) as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("body", ""),
                "url": item.get("href", ""),
                "source": item.get("source", ""),
            })
    return results


def _ddg_news(query: str, max_results: int = 8) -> list[dict]:
    """Retrieve news with DDGS, falling back to ordinary metasearch."""
    from ddgs import DDGS

    try:
        results = []
        with DDGS(timeout=8) as ddgs:
            for item in ddgs.news(query, max_results=max_results):
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("body", ""),
                    "url": item.get("url", ""),
                    "source": item.get("source", ""),
                    "date": item.get("date", ""),
                })
        if results:
            return results
    except Exception as exc:
        print(f"[WebSearch] DDGS news failed ({exc}) - falling back to text search")

    return _ddg_text(f"latest news {query}".strip(), max_results=max_results)


def _format_results(query: str, results: list[dict], heading: str = "Search results") -> str:
    if not results:
        return f"No results found for: {query}"

    lines = [f"{heading} for: {query}", ""]
    for index, item in enumerate(results, 1):
        title = item.get("title", "").strip()
        snippet = item.get("snippet", "").strip()
        url = item.get("url", "").strip()
        source = item.get("source", "").strip()
        date = item.get("date", "").strip()

        label = title or url or f"Result {index}"
        metadata = " | ".join(value for value in (source, date) if value)
        lines.append(f"{index}. {label}" + (f" [{metadata}]" if metadata else ""))
        if snippet:
            lines.append(f"   {snippet}")
        if url:
            lines.append(f"   Source: {url}")
        lines.append("")
    return "\n".join(lines).strip()


def _source_packet(results: list[dict]) -> str:
    blocks = []
    for index, item in enumerate(results, 1):
        blocks.append(
            "\n".join([
                f"SOURCE {index}",
                f"Title: {item.get('title', '')}",
                f"Publisher: {item.get('source', '')}",
                f"Date: {item.get('date', '')}",
                f"Snippet: {item.get('snippet', '')}",
                f"URL: {item.get('url', '')}",
            ])
        )
    return "\n\n".join(blocks)


def _synthesize(query: str, results: list[dict], instruction: str) -> str:
    """Use Flash-Lite only to synthesize already-retrieved free search results."""
    if not results:
        return f"No results found for: {query}"

    from google import genai

    prompt = f"""You are summarizing web search results for a desktop assistant.

User request:
{query}

Task:
{instruction}

Rules:
- Use only the supplied sources.
- Do not invent facts that are absent from the supplied sources.
- Preserve useful dates, prices, names, and qualifications.
- Cite claims inline as [1], [2], etc. using the source numbers below.
- End with a compact Sources section containing the source title and URL.
- Be direct and useful rather than discussing the search process.

SOURCES:
{_source_packet(results)}
"""

    client = genai.Client(api_key=_get_api_key())
    response = client.models.generate_content(
        model=HELPER_MODEL,
        contents=prompt,
    )
    text = _response_text(response)
    if not text:
        raise ValueError("Gemini returned an empty synthesis response.")
    return text


def _search(query: str) -> str:
    results = _ddg_text(query, max_results=8)
    try:
        return _synthesize(
            query,
            results,
            "Answer the request accurately and concisely from the strongest available results.",
        )
    except Exception as exc:
        print(f"[WebSearch] Gemini synthesis failed ({exc}) - returning raw results")
        return _format_results(query, results)


def _news(query: str) -> str:
    topic = query or "top world news today"
    results = _ddg_news(topic, max_results=10)
    try:
        return _synthesize(
            topic,
            results,
            "Summarize the latest significant developments. Prioritize explicit publication dates and avoid presenting undated items as breaking news.",
        )
    except Exception as exc:
        print(f"[WebSearch] Gemini news synthesis failed ({exc}) - returning raw results")
        return _format_results(topic, results, heading="Latest news")


def _research(query: str) -> str:
    results = _ddg_text(query, max_results=12)
    try:
        return _synthesize(
            query,
            results,
            "Provide a thorough explanation with background, key facts, current state, disagreements, and important caveats.",
        )
    except Exception as exc:
        print(f"[WebSearch] Gemini research synthesis failed ({exc}) - returning raw results")
        return _format_results(query, results, heading="Research sources")


def _price(query: str) -> str:
    search_query = f"current price of {query}"
    results = _ddg_text(search_query, max_results=8)
    try:
        return _synthesize(
            query,
            results,
            "Report current prices or price ranges, including retailer, configuration, region, and date where available. Do not merge incompatible variants into one price.",
        )
    except Exception as exc:
        print(f"[WebSearch] Gemini price synthesis failed ({exc}) - returning raw results")
        return _format_results(search_query, results, heading="Price results")


def _compare(items: list[str], aspect: str) -> str:
    query = f"Compare {', '.join(items)} for {aspect}"
    results: list[dict] = []
    for item in items:
        try:
            results.extend(_ddg_text(f"{item} {aspect}", max_results=4))
        except Exception as exc:
            print(f"[WebSearch] Search failed for {item!r}: {exc}")

    try:
        return _synthesize(
            query,
            results,
            f"Compare the requested items specifically on {aspect}. Clearly separate verified differences from missing information.",
        )
    except Exception as exc:
        print(f"[WebSearch] Gemini comparison synthesis failed ({exc}) - returning raw results")
        return _format_results(query, results, heading="Comparison sources")


def _gemini_headlines(n: int = 5) -> tuple[list[str], str]:
    """Compatibility helper for the startup briefing."""
    import re

    results = _ddg_news("top world news today", max_results=max(n + 3, 8))
    try:
        raw = _synthesize(
            "top world news today",
            results,
            f"Return exactly {n} major current headlines as a numbered list. Put only the headline on each numbered line, followed by a Sources section.",
        )
    except Exception as exc:
        print(f"[WebSearch] Gemini headline synthesis failed ({exc})")
        raw = _format_results("top world news today", results, heading="Latest news")

    headlines = []
    for line in raw.splitlines():
        match = re.match(r"^\s*\d+[.)-]\s*(.+?)\s*$", line)
        if not match:
            continue
        headline = re.sub(r"\s*\[\d+(?:,\s*\d+)*\]\s*$", "", match.group(1)).strip()
        if len(headline) > 10:
            headlines.append(headline)
        if len(headlines) >= n:
            break
    return headlines, raw.strip()


def web_search(
    parameters: dict,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    params = parameters or {}
    query = params.get("query", "").strip()
    mode = params.get("mode", "search").lower().strip()
    items = params.get("items", [])
    aspect = params.get("aspect", "general").strip() or "general"

    if not query and not items:
        return "Please provide a search query."

    if items and mode != "compare":
        mode = "compare"

    if player:
        player.write_log(f"[Search:{mode}] {query or ', '.join(items)}")

    print(f"[WebSearch] mode={mode!r} query={query!r}")

    try:
        if mode == "compare" and items:
            return _compare(items, aspect)
        if mode == "news":
            return _news(query)
        if mode == "research":
            return _research(query)
        if mode == "price":
            return _price(query)
        return _search(query)
    except Exception as exc:
        print(f"[WebSearch] All search backends failed: {exc}")
        return f"Search failed: {exc}"
