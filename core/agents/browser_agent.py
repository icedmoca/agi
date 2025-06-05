"""BrowserAgent – minimal headless page fetcher & scraper.

This agent focuses on retrieving publicly available web pages and
returning a cleaned textual representation.  It uses *requests* for
HTTP and falls back to *urllib* if necessary.  For HTML parsing a very
light-weight regex fallback is used when BeautifulSoup is unavailable
so the dependency remains optional.
"""

from __future__ import annotations

import logging
import re
from typing import List, Dict

try:
    import requests
except Exception:  # pragma: no cover – optional at runtime
    requests = None

logger = logging.getLogger(__name__)


class BrowserAgent:
    """Fetch a web page and return its readable text."""

    user_agent = "Mozilla/5.0 (compatible; BrowserAgent/0.1)"

    def _get(self, url: str) -> str:
        """Robust HTTP GET that works even when *requests* is missing."""
        headers = {"User-Agent": self.user_agent}
        if requests is not None:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            return resp.text
        # fallback – stdlib only
        from urllib.request import Request, urlopen

        req = Request(url, headers=headers)  # type: ignore[arg-type]
        with urlopen(req, timeout=20) as resp:  # type: ignore[arg-type]
            return resp.read().decode(errors="replace")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def summarize_html(self, html: str) -> str:
        """Generate a markdown-ish summary of *html*."""
        try:
            from bs4 import BeautifulSoup  # type: ignore

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
        except Exception:
            text = re.sub(r"<script[\s\S]*?</script>", "", html, flags=re.I)
            text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.I)
            text = re.sub(r"<[^>]+>", "", text)

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        snippet = "\n".join(lines[:60])  # first ~60 lines
        return snippet

    def browse_page(self, url: str) -> Dict[str, str]:
        """Return dict with raw HTML and markdown summary."""
        try:
            html = self._get(url)
            summary = self.summarize_html(html)
            return {"raw_html": html[:20000], "markdown_summary": summary}
        except Exception as e:
            logger.error("browse_page error: %s", e)
            return {"error": str(e)} 