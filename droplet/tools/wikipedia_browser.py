"""Wikipedia browser tool for searching and reading Wikipedia articles"""

import html
import logging
import time

import wikipedia
from gpt_oss.tools.simple_browser.backend import BackendError
from gpt_oss.tools.simple_browser.page_contents import process_html
from gpt_oss.tools.simple_browser.simple_browser_tool import SimpleBrowserTool
from openai_harmony import ToolNamespaceConfig

logger = logging.getLogger(__name__)


class WikipediaBackend:
    """
    Backend for searching and browsing Wikipedia articles.
    """

    source = "wikipedia"

    def __init__(self, language="en"):
        wikipedia.set_lang(language)
        self.language = language

    async def search(self, query, topn=10, session=None):
        """
        Search Wikipedia for articles matching the query.
        """
        search_results = wikipedia.search(query, results=topn)

        if not search_results:
            raise BackendError(f"No Wikipedia results found for query: '{query}'")

        title_url_summary = []
        for title in search_results:
            try:
                summary = wikipedia.summary(title, sentences=2, auto_suggest=False)
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                title_url_summary.append((
                    html.escape(title, quote=True),
                    html.escape(url, quote=True),
                    html.escape(summary, quote=True)
                ))
            except wikipedia.exceptions.DisambiguationError:
                continue
            except wikipedia.exceptions.PageError:
                continue
            except Exception as e:
                logger.warning(f"Error getting summary for '{title}': {e}")
                continue

        if not title_url_summary:
            raise BackendError(f"Could not retrieve summaries for any Wikipedia results for: '{query}'")

        html_page = f"""
<html>
<head><title>Wikipedia Search: {html.escape(query)}</title></head>
<body>
<h1>Wikipedia Search Results: {html.escape(query)}</h1>
<p>Found {len(title_url_summary)} articles</p>
<ul>
{"".join([f"<li><strong>{title}</strong><br/><p>{summary}</p><p><em>URL: {url}</em></p></li>" for title, url, summary in title_url_summary])}
</ul>
</body>
</html>
"""

        pseudo_url = f"wikipedia-search://{query}?ts={int(time.time())}"

        return process_html(
            html=html_page,
            url=pseudo_url,
            title=f"Wikipedia: {query}",
            display_urls=False,
            session=session,
        )

    async def fetch(self, url, session=None):
        """
        Fetch Wikipedia article content.
        """
        if url.startswith("wikipedia-search://") or url.startswith("wikipedia-disambiguation://"):
            raise BackendError(
                "Cannot open search results page directly. "
                "Please use the 'open' function with a specific Wikipedia article URL from the search results."
            )

        if "wikipedia.org/wiki/" in url:
            title = url.split("/wiki/")[-1].replace("_", " ")
        else:
            title = url

        page = wikipedia.page(title, auto_suggest=False)

        if not hasattr(page, 'url') or not page.url:
            page.url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

        content_html = self._format_wikipedia_content(page)

        return process_html(
            html=content_html,
            url=page.url,
            title=page.title,
            display_urls=False,
            session=session,
        )

    def _format_wikipedia_content(self, page):
        content_paragraphs = page.content.split('\n\n')
        formatted_content = ""

        for para in content_paragraphs:
            para = para.strip()
            if not para:
                continue

            if para.isupper() or (len(para) < 100 and para.endswith(":")):
                formatted_content += f"<h2>{html.escape(para)}</h2>\n"
            else:
                formatted_content += f"<p>{html.escape(para)}</p>\n"

        html_page = f"""
<html>
<head><title>{html.escape(page.title)}</title></head>
<body>
<h1>{html.escape(page.title)}</h1>
<p><strong>Source:</strong> {html.escape(page.url)}</p>
<hr/>
{formatted_content}
<hr/>
<h3>References</h3>
<ul>
{"".join([f"<li>{html.escape(ref)}</li>" for ref in page.references[:10]])}
</ul>
</body>
</html>
"""
        return html_page


class WikipediaBrowserTool(SimpleBrowserTool):
    """
    Browser tool for searching and browsing Wikipedia articles.

    Uses the "wikipedia" namespace with standard SimpleBrowserTool functions.
    Just wraps a WikipediaBackend that formats Wikipedia data as HTML for browsing.

    Provides:
    - search: Search for Wikipedia articles
    - open: Read Wikipedia article content
    - find: Find text patterns in the current page
    """

    def __init__(self, language="en"):
        # SimpleBrowserTool asserts name=="browser", so we pass that
        # and override the name property below
        super().__init__(backend=WikipediaBackend(language=language), name="browser")

    @classmethod
    def get_tool_name(cls) -> str:
        return "wikipedia"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def tool_config(self):
        # Use native gpt-oss browser config structure but customize descriptions for Wikipedia
        config = ToolNamespaceConfig.browser()
        config.name = "wikipedia"
        config.description = """Tool for browsing Wikipedia.
The `cursor` appears in brackets before each browsing display: `[{cursor}]`.
Use search(query="topic") to search for Wikipedia articles.
Use open(id=N) to read an article from search results, where N is the link number.
Cite information from the tool using the following format:
`【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.
Do not quote more than 10 words directly from the tool output.
sources=""" + self.backend.source

        # Customize the search function description for Wikipedia
        # Convert to dict to modify
        config_dict = config.model_dump()
        for tool_def in config_dict['tools']:
            if tool_def['name'] == 'search':
                tool_def['description'] = 'Search Wikipedia for articles matching a query.'
                tool_def['parameters']['properties']['query']['description'] = 'Search term for Wikipedia articles'

        # Recreate config from modified dict
        config = ToolNamespaceConfig(**config_dict)

        return config
