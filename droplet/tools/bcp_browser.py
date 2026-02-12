import html
import json
import logging
import time
from typing import AsyncIterator

from aiohttp import ClientSession
from gpt_oss.tools.simple_browser.backend import BackendError
from gpt_oss.tools.simple_browser.page_contents import process_html
from gpt_oss.tools.simple_browser.simple_browser_tool import (
    SimpleBrowserTool, maybe_get_function_args)
from openai_harmony import (Author, Message, Role, TextContent,
                            ToolNamespaceConfig)

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

        self.retrieved_docs = {}

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
        endpoint = f"{self.base_url}/search"
        payload = {"query": query, "topn": int(topn)}

        needs_cleanup = False
        if session is None:
            session = ClientSession()
            needs_cleanup = True

        async with session.post(endpoint, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                if needs_cleanup:
                    await session.close()
                raise BackendError(f"BCP search error {resp.status}: {error_text}")

            data = await resp.json()
            results = data.get("results", [])

        if needs_cleanup:
            await session.close()

        if not results:
            raise BackendError(f"No results returned for query: '{query}'")

        title_url_summary = []
        self.retrieved_docs = {}

        for idx, result in enumerate(results):
            title = result.get('title', 'No title')
            url = result.get('url', f'doc_{idx}')
            summary = result.get('summary', '')

            title_url_summary.append((
                html.escape(title, quote=True),
                url,
                html.escape(summary, quote=True)
            ))

            self.retrieved_docs[url] = {
                'title': html.escape(title, quote=True),
                'summary': html.escape(summary, quote=True)
            }

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
        Fetch a previously retrieved document by its URL.

        Args:
            url: Document URL (from search results)
            session: Optional aiohttp session

        Returns:
            Processed HTML page with document content
        """
        data = self.retrieved_docs.get(url, None)

        if not data:
            raise BackendError(f"No content returned for {url}")

        content = data.get("summary", "")
        if not content:
            raise BackendError(f"No content available for {url}")

        return process_html(
            html=content,
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
        super().__init__(backend=backend)
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
        config.name = "bcp"
        config.description = """Tool for searching the BrowseComp-Plus corpus. If searching, you must ask for no more than 5 search results.
        The `cursor` appears in brackets before each browsing display: `[{cursor}]`.
        Cite information from the tool using the following format:
        `【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.
        Do not quote more than 10 words directly from the tool output.
        sources=""" + self.backend.source

        return config
