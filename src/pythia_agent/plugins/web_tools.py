"""Web tools plugin: HTTP requests, web search (Exa, Tavily), RSS feeds."""

import json
import logging
import os
from typing import Any

import requests

from strands import tool
from strands.plugins import Plugin

logger = logging.getLogger(__name__)


GUIDANCE = (
    "\n\nYou can search the web and fetch URLs. Use exa_search or tavily_search for research, "
    "http_request for direct API calls, and rss_read for monitoring feeds."
)


class WebToolsPlugin(Plugin):
    """Provides HTTP request, web search, and RSS feed tools."""

    name = "web-tools"

    def __init__(self, request_timeout: int = 30):
        self.request_timeout = request_timeout
        super().__init__()

    def init_agent(self, agent) -> None:
        agent.system_prompt += GUIDANCE

    @tool
    def http_request(
        self,
        url: str,
        method: str = "GET",
        headers: str = "",
        body: str = "",
        timeout: int = 0,
    ) -> str:
        """Make an HTTP request and return the response.

        Args:
            url: The URL to request
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            headers: JSON string of headers to include
            body: Request body (for POST/PUT/PATCH)
            timeout: Request timeout in seconds (0 uses default)
        """
        t = timeout or self.request_timeout
        hdrs = {}
        if headers:
            try:
                hdrs = json.loads(headers)
            except json.JSONDecodeError:
                return "Error: headers must be valid JSON"

        try:
            resp = requests.request(
                method=method.upper(),
                url=url,
                headers=hdrs,
                data=body if body else None,
                timeout=t,
            )
            content_type = resp.headers.get("content-type", "")
            body_text = resp.text[:10000]
            if len(resp.text) > 10000:
                body_text += f"\n... (truncated, total {len(resp.text)} chars)"

            return f"Status: {resp.status_code}\nContent-Type: {content_type}\n\n{body_text}"
        except requests.Timeout:
            return f"Error: request timed out after {t}s"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def exa_search(
        self,
        query: str,
        num_results: int = 5,
        include_text: bool = True,
        include_domains: str = "",
        exclude_domains: str = "",
    ) -> str:
        """Search the web using Exa for high-quality, AI-optimized results.

        Requires EXA_API_KEY environment variable.

        Args:
            query: Search query
            num_results: Number of results to return (1-10)
            include_text: Whether to include full page text content
            include_domains: Comma-separated list of domains to restrict to
            exclude_domains: Comma-separated list of domains to exclude
        """
        api_key = os.environ.get("EXA_API_KEY")
        if not api_key:
            return "Error: EXA_API_KEY environment variable not set"

        payload: dict[str, Any] = {
            "query": query,
            "numResults": min(num_results, 10),
            "contents": {"text": include_text},
        }
        if include_domains:
            payload["includeDomains"] = [d.strip() for d in include_domains.split(",")]
        if exclude_domains:
            payload["excludeDomains"] = [d.strip() for d in exclude_domains.split(",")]

        try:
            resp = requests.post(
                "https://api.exa.ai/search",
                headers={"x-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=self.request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for r in data.get("results", []):
                entry = f"**{r.get('title', 'Untitled')}**\n{r.get('url', '')}"
                if include_text and r.get("text"):
                    entry += f"\n{r['text'][:1000]}"
                results.append(entry)

            return "\n\n---\n\n".join(results) if results else "No results found."
        except Exception as e:
            return f"Error: {e}"

    @tool
    def tavily_search(
        self,
        query: str,
        num_results: int = 5,
        search_depth: str = "basic",
        include_domains: str = "",
        exclude_domains: str = "",
    ) -> str:
        """Search the web using Tavily, optimized for AI agent retrieval.

        Requires TAVILY_API_KEY environment variable.

        Args:
            query: Search query
            num_results: Number of results (1-10)
            search_depth: 'basic' for fast results, 'advanced' for deeper search
            include_domains: Comma-separated domains to restrict to
            exclude_domains: Comma-separated domains to exclude
        """
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return "Error: TAVILY_API_KEY environment variable not set"

        payload: dict[str, Any] = {
            "api_key": api_key,
            "query": query,
            "max_results": min(num_results, 10),
            "search_depth": search_depth,
        }
        if include_domains:
            payload["include_domains"] = [d.strip() for d in include_domains.split(",")]
        if exclude_domains:
            payload["exclude_domains"] = [d.strip() for d in exclude_domains.split(",")]

        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json=payload,
                timeout=self.request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for r in data.get("results", []):
                entry = f"**{r.get('title', 'Untitled')}**\n{r.get('url', '')}\n{r.get('content', '')[:800]}"
                results.append(entry)

            return "\n\n---\n\n".join(results) if results else "No results found."
        except Exception as e:
            return f"Error: {e}"

    @tool
    def rss_read(self, url: str, max_entries: int = 10) -> str:
        """Fetch and parse an RSS/Atom feed.

        Args:
            url: URL of the RSS or Atom feed
            max_entries: Maximum number of entries to return
        """
        try:
            import feedparser
        except ImportError:
            return "Error: feedparser not installed (pip install feedparser)"

        try:
            feed = feedparser.parse(url)
            if feed.bozo and not feed.entries:
                return f"Error parsing feed: {feed.bozo_exception}"

            entries = []
            for entry in feed.entries[:max_entries]:
                title = entry.get("title", "Untitled")
                link = entry.get("link", "")
                published = entry.get("published", "")
                summary = entry.get("summary", "")[:300]
                entries.append(f"**{title}**\n{link}\n{published}\n{summary}")

            header = f"Feed: {feed.feed.get('title', url)} ({len(feed.entries)} total entries)\n\n"
            return header + "\n\n---\n\n".join(entries) if entries else "No entries in feed."
        except Exception as e:
            return f"Error: {e}"
