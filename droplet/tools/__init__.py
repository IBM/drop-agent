"""Droplet tools for agent interactions"""

# Import base utilities
from droplet.tools.base import (SimpleFunctionTool,
                                convert_tool_config_to_openai)
# Import file browser
from droplet.tools.file_browser import FileBackend, FileBrowserTool
# Import Milvus tools
from droplet.tools.milvus_retriever import MilvusBackend, RetrieverBrowserTool
# Import Python execution tool
from droplet.tools.python_tool import PythonTool
# Import Semantic Scholar
from droplet.tools.semantic_scholar import (SemanticScholarBackend,
                                            SemanticScholarTool)
# Import Wikipedia browser
from droplet.tools.wikipedia_browser import (WikipediaBackend,
                                             WikipediaBrowserTool)
# Import BCP browser
from droplet.tools.bcp_browser import BCPBackend, BCPBrowserTool

# Maintain BrowseTool alias (deprecated)
BrowseTool = FileBrowserTool

__all__ = [
    # Base
    'SimpleFunctionTool',
    'convert_tool_config_to_openai',
    # File browser
    'FileBackend',
    'FileBrowserTool',
    # Wikipedia
    'WikipediaBackend',
    'WikipediaBrowserTool',
    # BCP browser
    'BCPBackend',
    'BCPBrowserTool',
    # Semantic Scholar
    'SemanticScholarBackend',
    'SemanticScholarTool',
    # Milvus
    'MilvusBackend',
    'RetrieverBrowserTool',
    # Python execution
    'PythonTool',
    # Deprecated
    'BrowseTool',
]
