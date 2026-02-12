"""
Browser tool for retriever with a local Milvus vector database backend.
Adapted from gma-rag-rl/src/retrievers/browser_retriever_tool.py
"""

import html
import json
import logging
import time
from typing import AsyncIterator

from gpt_oss.tools.simple_browser.backend import BackendError
from gpt_oss.tools.simple_browser.page_contents import process_html
from gpt_oss.tools.simple_browser.simple_browser_tool import (
    SimpleBrowserTool, maybe_get_function_args)
from openai_harmony import (Author, Message, Role, TextContent,
                            ToolNamespaceConfig)

logger = logging.getLogger(__name__)


class MilvusBackend:
    """
    Backend for searching a local Milvus vector database.

    Uses SentenceTransformers for local embedding computation and returns results as HTML.
    """
    source = "local database"

    def __init__(self, model_name_or_path, milvus_filename, milvus_collection_name="nq_granite125m_512_100_20250530"):
        """
        Initialize with a local embedding model and Milvus database file.

        Args:
            model_name_or_path: Path or name of the SentenceTransformer model
            milvus_filename: Path to the local Milvus database file
            milvus_collection_name: The name of the Milvus collection to use
        """
        from pymilvus import MilvusClient
        from sentence_transformers import SentenceTransformer

        if milvus_collection_name.find("-") >= 0:
            milvus_collection_name = milvus_collection_name.replace("-", "_")

        print(f'Loading SentenceTransformer model: {model_name_or_path}')
        self.model = SentenceTransformer(model_name_or_path, device='cpu')

        self.milvus_client = MilvusClient(uri=milvus_filename)
        self.milvus_collection_name = milvus_collection_name
        print(f'Initialized Milvus backend with database: {milvus_filename}')

        self.retrieved_docs = {}

    def _compute_embedding(self, text):
        """
        Compute embedding for text using local SentenceTransformer model.

        Args:
            text: String or list of strings to embed

        Returns:
            Embedding vector or list of vectors
        """
        if isinstance(text, str):
            text = [text]

        query_vector = self.model.encode(text, show_progress_bar=False).tolist()

        return query_vector[0] if len(query_vector) == 1 else query_vector

    def _search_milvus(self, query, top_k):
        """
        Perform search using a query string (automatically computes embeddings).

        Args:
            query: String query to search for
            top_k: Number of results to return

        Returns:
            List of search results with title, text, and score
        """
        query_embedding = self._compute_embedding(query)

        res = self.milvus_client.search(
            collection_name=self.milvus_collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["title", "text"],
        )

        documents = []
        if isinstance(res, list) and isinstance(res[0], list):
            res = res[0]

        for document in res:
            if 'entity' in document:
                document['entity']['score'] = document['distance']
                documents.append(document['entity'])
            else:
                document['score'] = document['distance']
                documents.append(document)

        return documents

    async def search(
        self,
        query: str,
        topn: int = 3,
        session = None,
    ):
        """
        Search for documents matching the query.

        Args:
            query: Search query string
            topn: Number of top results to return
            session: Optional aiohttp session

        Returns:
            Processed HTML page with search results
        """
        results = self._search_milvus(query=query, top_k=int(topn))

        if not results:
            raise BackendError(f"No results returned for query: '{query}'")

        title_url_summary = []
        # Create an empty dict after every search
        self.retrieved_docs = {}

        for idx, result in enumerate(results):
            # Create a dummy url
            doc_url = f"doc_{str(idx)}_{'_'.join(result['title'].lower().split(' '))}"

            title_url_summary.append((
                html.escape(result['title'], quote=True),
                doc_url,
                html.escape(result['text'], quote=True)
            ))

            self.retrieved_docs[doc_url] = {
                'text': html.escape(result['text'], quote=True),
                'title': html.escape(result['title'], quote=True)
            }

        html_page = f"""
<html><body>
<h1>Search Results</h1>
<ul>
{"".join([f"<li><a href='{url}'>{title}</a> {summary}</li>" for title, url, summary in title_url_summary])}
</ul>
</body></html>
"""

        pseudo_url = f"local-search://ts={int(time.time())}"
        return process_html(
            html=html_page,
            url=pseudo_url,
            title=query,
            display_urls=True,
            session=session,
        )

    async def fetch(self, url: str, session=None):
        """
        Fetch a previously retrieved document by its URL.

        Args:
            url: Document URL (from search results)
            session: Optional aiohttp session

        Returns:
            Processed HTML page with document content
        """
        data = self.retrieved_docs.get(url, None)

        if not data or not data.get("text"):
            raise BackendError(f"No content returned for {url}")

        return process_html(
            html=data.get("text", ""),
            url=url,
            title=data.get("title", ""),
            display_urls=True,
            session=session,
        )


class RetrieverBrowserTool(SimpleBrowserTool):
    """
    Browser tool for searching and retrieving documents from a local Milvus vector database.

    Uses the "retriever" namespace with standard SimpleBrowserTool functions.
    Wraps a MilvusBackend that formats retrieval results as HTML for browsing.

    Provides:
    - search: Search for documents by keywords
    - open: Get detailed information about a specific document
    - find: Find text patterns in the current page
    """

    def __init__(self, milvus_db=None, milvus_model=None, milvus_collection=None):
        """
        Initialize RetrieverBrowserTool with Milvus configuration.

        Args:
            milvus_db: Path to Milvus database file (required)
            milvus_model: SentenceTransformer model name or path
            milvus_collection: Milvus collection name
        """
        import os

        if not milvus_db:
            raise ValueError("milvus_db is required for RetrieverBrowserTool")

        # Expand ~ in path
        db_path = os.path.expanduser(milvus_db)

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Milvus database not found: {db_path}")

        print(f"Loading Milvus database: {db_path}")
        print(f"  Model: {milvus_model}")
        print(f"  Collection: {milvus_collection}\n")

        backend = MilvusBackend(
            model_name_or_path=milvus_model,
            milvus_filename=db_path,
            milvus_collection_name=milvus_collection
        )
        super().__init__(backend=backend)
        print("✓ Milvus retriever loaded successfully\n")

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        """Process tool messages, handling browser.retriever.search format."""
        # Handle both "browser.search" and "browser.retriever.search" formats
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
        config.name = "retriever"
        config.description = """Tool for browsing a local document collection. If searching, you must ask for no more than 5 search results.
        The `cursor` appears in brackets before each browsing display: `[{cursor}]`.
        Cite information from the tool using the following format:
        `【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.
        Do not quote more than 10 words directly from the tool output.
        sources=""" + self.backend.source

        return config
