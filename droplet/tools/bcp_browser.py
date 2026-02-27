import html
import logging
import time
from typing import AsyncIterator

from aiohttp import ClientSession
from gpt_oss.tools.simple_browser.backend import (VIEW_SOURCE_PREFIX,
                                                  BackendError)
from gpt_oss.tools.simple_browser.page_contents import process_html
from gpt_oss.tools.simple_browser.simple_browser_tool import SimpleBrowserTool
from openai_harmony import Message, ToolNamespaceConfig

logger = logging.getLogger(__name__)


class BCPBackend:
    """
    Backend for searching the BrowseComp-Plus corpus.

    Connects to a BCP search service (uvicorn server) and returns results as HTML.
    """
    source = "BrowseComp-Plus corpus"

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize with BCP server URL.

        Args:
            base_url: Base URL of the BCP search service
        """
        base_url = base_url.rstrip('/')
        if not base_url.startswith(('http://', 'https://')):
            base_url = f'http://{base_url}'
        self.base_url = base_url
        logger.info(f"Initialized BCPBackend with base_url={self.base_url}")

    async def _post(self, session: ClientSession, endpoint: str, payload: dict) -> dict:
        async with session.post(f"{self.base_url}{endpoint}", json=payload) as resp:
            if resp.status != 200:
                raise BackendError(f"BCP error {resp.status}: {await resp.text()}")
            return await resp.json()

    async def search(
        self,
        query: str,
        topn: int = 5,
        session: ClientSession = None,
    ):
        """
        Search the BCP corpus for documents matching the query.

        Args:
            query: Search query string
            topn: Number of top results to return
            session: Optional aiohttp session

        Returns:
            Processed HTML page with search results
        """
        needs_cleanup = False
        if session is None:
            session = ClientSession()
            needs_cleanup = True

        try:
            data = await self._post(session, "/search", {"query": query, "topn": int(topn)})
        finally:
            if needs_cleanup:
                await session.close()

        results = data.get("results", [])

        if not results:
            raise BackendError(f"No results returned for query: '{query}'")

        title_url_summary = []

        for idx, result in enumerate(results):
            title = result.get('title', 'No title')
            url = result.get('url', f'doc_{idx}')
            summary = result.get('summary', '')

            title_url_summary.append((
                html.escape(title, quote=True),
                html.escape(url, quote=True),
                html.escape(summary, quote=True)
            ))

        html_page = f"""
<html><body>
<h1>Search Results</h1>
<ul>
{"".join([f"<li><a href='{url}'>{title}</a> {summary}</li>" for title, url, summary in title_url_summary])}
</ul>
</body></html>
"""

        pseudo_url = f"bcp-search://ts={int(time.time())}"
        return process_html(
            html=html_page,
            url=pseudo_url,
            title=query,
            display_urls=True,
            session=session,
        )

    async def fetch(self, url: str, session: ClientSession = None):
        """
        Fetch a document by URL from the BCP server.

        Args:
            url: Document URL (from search results)
            session: Optional aiohttp session

        Returns:
            Processed HTML page with document content
        """
        is_view_source = url.startswith(VIEW_SOURCE_PREFIX)
        if is_view_source:
            url = url[len(VIEW_SOURCE_PREFIX):]

        needs_cleanup = False
        if session is None:
            session = ClientSession()
            needs_cleanup = True

        try:
            data = await self._post(session, "/get_content", {"url": url})
        finally:
            if needs_cleanup:
                await session.close()

        if not data or not data.get("content"):
            raise BackendError(f"No content returned for {url}")

        return process_html(
            html=data.get("content", ""),
            url=url,
            title=data.get("title", ""),
            display_urls=True,
            session=session,
        )


class BCPBrowserTool(SimpleBrowserTool):
    """
    Browser tool for searching the BrowseComp-Plus corpus.

    Uses the "bcp" namespace with standard SimpleBrowserTool functions.
    Wraps a BCPBackend that formats search results as HTML for browsing.

    Provides:
    - search: Search for documents by keywords
    - open: Get detailed information about a specific document
    - find: Find text patterns in the current page
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize BCPBrowserTool with server URL.

        Args:
            base_url: Base URL of the BCP search service
        """
        backend = BCPBackend(base_url=base_url)
        super().__init__(backend=backend, max_search_results=5)
        logger.info("✓ BCP browser tool loaded successfully")

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        """Process tool messages, handling browser.bcp.search format."""
        # Handle both "browser.search" and "browser.bcp.search" formats
        # Extract the actual function name (last part after splitting by ".")
        parts = message.recipient.split(".")
        function_name = parts[-1]

        # Reconstruct as browser.{function_name} for parent class
        reconstructed_recipient = f"browser.{function_name}"

        # Create a new message with the corrected recipient
        corrected_message = Message(
            author=message.author,
            content=message.content,
            channel=message.channel
        ).with_recipient(reconstructed_recipient)

        # Call parent's _process with corrected message
        async for msg in super()._process(corrected_message):
            yield msg

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        """Get tool configuration."""
        config = ToolNamespaceConfig.browser()
        config.name = "browser"
        config.description = """Tool for browsing.
The `cursor` appears in brackets before each browsing display: `[{cursor}]`.
Cite information from the tool using the following format:
`【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.
Do not quote more than 10 words directly from the tool output.
sources=web"""

        return config
