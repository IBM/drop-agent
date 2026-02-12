"""Semantic Scholar tool for searching and retrieving scientific papers"""

import html
import time

from gpt_oss.tools.simple_browser.backend import BackendError
from gpt_oss.tools.simple_browser.page_contents import process_html
from gpt_oss.tools.simple_browser.simple_browser_tool import SimpleBrowserTool
from openai_harmony import ToolNamespaceConfig


class SemanticScholarBackend:
    """
    Backend for searching and retrieving scientific papers from Semantic Scholar.

    Rate limiting:
    - Without API key: 1 request per second (shared pool is 1000/s but we're conservative)
    - With API key: 10 requests per second (authenticated users get higher limits)

    API documentation: https://api.semanticscholar.org/api-docs/graph
    """

    source = "papers"
    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    # Fields to request for paper search results
    SEARCH_FIELDS = "paperId,title,abstract,year,citationCount,authors,venue,openAccessPdf,publicationDate"

    # Fields to request for paper details
    DETAIL_FIELDS = "paperId,title,abstract,year,citationCount,authors,venue,openAccessPdf,publicationDate,references,citations"

    def __init__(self, api_key=None, max_chars=32000):
        self.api_key = api_key
        self.max_chars = max_chars
        self._last_request_time = 0
        # Rate limit: 1 req/s without key, 10 req/s with key
        self._min_request_interval = 0.1 if api_key else 1.0

    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _get_headers(self):
        """Get request headers including API key if available"""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _make_request(self, url, params=None):
        """Make a rate-limited request to the Semantic Scholar API"""
        import requests

        self._rate_limit()
        response = requests.get(url, params=params, headers=self._get_headers(), timeout=30)
        response.raise_for_status()
        return response.json()

    async def search(self, query, topn=10, year=None, fields_of_study=None, session=None):
        """
        Search for papers using the bulk search endpoint.

        Args:
            query: Search query (supports boolean operators: +, |, -, "", *)
            topn: Maximum number of results (default: 10, max: 100)
            year: Filter by publication year (e.g., "2020" or "2020-2023")
            fields_of_study: Filter by field (e.g., "Computer Science")

        Returns:
            Formatted HTML page with search results
        """
        import requests

        url = f"{self.BASE_URL}/paper/search/bulk"

        params = {
            "query": query,
            "fields": self.SEARCH_FIELDS,
            "limit": min(topn, 100),
        }

        if year:
            params["year"] = year
        if fields_of_study:
            params["fieldsOfStudy"] = fields_of_study

        self._rate_limit()
        response = requests.get(url, params=params, headers=self._get_headers(), timeout=30)
        response.raise_for_status()
        data = response.json()

        papers = data.get("data", [])

        if not papers:
            raise BackendError(f"No papers found for query: '{query}'")

        # Build HTML results
        results_html = []
        for paper in papers:
            paper_id = paper.get("paperId", "")
            title = html.escape(paper.get("title", "No title"))
            abstract = paper.get("abstract", "")
            if abstract:
                abstract = html.escape(abstract[:500] + "..." if len(abstract) > 500 else abstract)
            else:
                abstract = "<em>No abstract available</em>"

            year_str = paper.get("year", "Unknown")
            citations = paper.get("citationCount", 0)
            venue = html.escape(paper.get("venue", "") or "")

            authors_list = paper.get("authors", [])
            if authors_list:
                author_names = [a.get("name", "") for a in authors_list[:5]]
                authors_str = html.escape(", ".join(author_names))
                if len(authors_list) > 5:
                    authors_str += f" et al. ({len(authors_list)} authors)"
            else:
                authors_str = "Unknown authors"

            # Check for open access PDF
            pdf_info = paper.get("openAccessPdf")
            pdf_link = ""
            if pdf_info and pdf_info.get("url"):
                pdf_url = html.escape(pdf_info["url"])
                pdf_link = f' | <a href="{pdf_url}">[PDF]</a>'

            # Create a link for the paper so process_html can extract it
            results_html.append(f"""
<li>
    <strong><a href="{paper_id}">{title}</a></strong><br/>
    <em>{authors_str}</em><br/>
    <small>{venue} ({year_str}) | Citations: {citations}{pdf_link}</small><br/>
    <p>{abstract}</p>
    <p><em>Paper ID: {paper_id}</em></p>
</li>
""")

        html_page = f"""
<html>
<head><title>Semantic Scholar Search: {html.escape(query)}</title></head>
<body>
<h1>Semantic Scholar Search Results: {html.escape(query)}</h1>
<p>Found {len(papers)} papers</p>
<ul>
{"".join(results_html)}
</ul>
</body>
</html>
"""

        pseudo_url = f"semantic-scholar-search://{query}?ts={int(time.time())}"

        return process_html(
            html=html_page,
            url=pseudo_url,
            title=f"Semantic Scholar: {query}",
            display_urls=False,
            session=session,
        )

    async def fetch(self, paper_id, session=None):
        """
        Fetch detailed information about a specific paper.

        Args:
            paper_id: Semantic Scholar paper ID, DOI, ArXiv ID, etc.
                     Formats: S2PaperId, CorpusId:123, DOI:10.xxx, ARXIV:xxx, etc.

        Returns:
            Formatted HTML page with paper details
        """
        import requests

        # Handle pseudo-URLs from search results
        if paper_id.startswith("semantic-scholar-search://"):
            raise BackendError(
                "Cannot open search results page directly. "
                "Please use the 'open' function with a specific paper ID from the search results."
            )

        url = f"{self.BASE_URL}/paper/{paper_id}"
        params = {"fields": self.DETAIL_FIELDS}

        self._rate_limit()
        response = requests.get(url, params=params, headers=self._get_headers(), timeout=30)

        if response.status_code == 404:
            raise BackendError(f"Paper not found: {paper_id}")
        response.raise_for_status()

        paper = response.json()

        # Extract paper information
        title = html.escape(paper.get("title", "No title"))
        abstract = paper.get("abstract", "")
        if abstract:
            abstract = html.escape(abstract)
        else:
            abstract = "<em>No abstract available</em>"

        year_str = paper.get("year", "Unknown")
        citations = paper.get("citationCount", 0)
        venue = html.escape(paper.get("venue", "") or "Unknown venue")
        pub_date = paper.get("publicationDate", "")

        authors_list = paper.get("authors", [])
        if authors_list:
            authors_html = "<ul>"
            for author in authors_list:
                author_name = html.escape(author.get("name", "Unknown"))
                author_id = author.get("authorId", "")
                authors_html += f"<li>{author_name} (ID: {author_id})</li>"
            authors_html += "</ul>"
        else:
            authors_html = "<p>Unknown authors</p>"

        # Open access PDF
        pdf_info = paper.get("openAccessPdf")
        pdf_html = ""
        if pdf_info and pdf_info.get("url"):
            pdf_url = html.escape(pdf_info["url"])
            pdf_html = f'<p><strong>Open Access PDF:</strong> <a href="{pdf_url}">{pdf_url}</a></p>'

        # References (limit to first 20)
        references = paper.get("references", [])
        refs_html = ""
        if references:
            refs_html = "<h2>References (first 20)</h2><ul>"
            for ref in references[:20]:
                ref_title = html.escape(ref.get("title", "Unknown title") or "Unknown title")
                ref_id = ref.get("paperId", "")
                refs_html += f"<li>{ref_title} (ID: {ref_id})</li>"
            refs_html += "</ul>"
            if len(references) > 20:
                refs_html += f"<p><em>...and {len(references) - 20} more references</em></p>"

        # Citations (limit to first 20)
        citations_list = paper.get("citations", [])
        cites_html = ""
        if citations_list:
            cites_html = "<h2>Cited By (first 20)</h2><ul>"
            for cite in citations_list[:20]:
                cite_title = html.escape(cite.get("title", "Unknown title") or "Unknown title")
                cite_id = cite.get("paperId", "")
                cites_html += f"<li>{cite_title} (ID: {cite_id})</li>"
            cites_html += "</ul>"
            if len(citations_list) > 20:
                cites_html += f"<p><em>...and {len(citations_list) - 20} more citations</em></p>"

        s2_paper_id = paper.get("paperId", paper_id)
        s2_url = f"https://www.semanticscholar.org/paper/{s2_paper_id}"

        html_page = f"""
<html>
<head><title>{title}</title></head>
<body>
<h1>{title}</h1>
<p><strong>Venue:</strong> {venue} ({year_str})</p>
<p><strong>Publication Date:</strong> {pub_date}</p>
<p><strong>Citations:</strong> {citations}</p>
<p><strong>Semantic Scholar URL:</strong> <a href="{s2_url}">{s2_url}</a></p>
{pdf_html}
<h2>Authors</h2>
{authors_html}
<h2>Abstract</h2>
<p>{abstract}</p>
{refs_html}
{cites_html}
</body>
</html>
"""

        return process_html(
            html=html_page,
            url=s2_url,
            title=title,
            display_urls=False,
            session=session,
        )


class SemanticScholarTool(SimpleBrowserTool):
    """
    Browser tool for searching and retrieving scientific papers from Semantic Scholar.

    Uses the "papers" namespace with standard SimpleBrowserTool functions.
    Just wraps a SemanticScholarBackend that formats paper data as HTML for browsing.

    Provides:
    - search: Search for papers by keywords with optional filters
    - open: Get detailed information about a specific paper
    - find: Find text patterns in the current page
    """

    def __init__(self, api_key=None):
        # SimpleBrowserTool asserts name=="browser", so we pass that
        # and override the name property below
        super().__init__(backend=SemanticScholarBackend(api_key=api_key), name="browser")

    @classmethod
    def get_tool_name(cls) -> str:
        return "papers"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def tool_config(self):
        # Use native gpt-oss browser config structure but customize descriptions for papers
        config = ToolNamespaceConfig.browser()
        config.name = "papers"
        config.description = """Tool for browsing scientific papers on Semantic Scholar.
The `cursor` appears in brackets before each browsing display: `[{cursor}]`.
Use search(query="keywords") to search for papers. Supports boolean operators: + (AND), | (OR), - (NOT).
Use open(id=N) to read paper details from search results, where N is the link number.
Cite information from the tool using the following format:
`【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.
Do not quote more than 10 words directly from the tool output.
sources=""" + self.backend.source

        # Customize the search function description for papers
        # Convert to dict to modify
        config_dict = config.model_dump()
        for tool_def in config_dict['tools']:
            if tool_def['name'] == 'search':
                tool_def['description'] = 'Search Semantic Scholar for papers. Supports boolean operators: +term (must have), -term (must not have), term1 | term2 (OR).'
                tool_def['parameters']['properties']['query']['description'] = 'Search query with optional boolean operators (+, -, |)'

        # Recreate config from modified dict
        config = ToolNamespaceConfig(**config_dict)

        return config
