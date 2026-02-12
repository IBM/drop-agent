import json
import re
import sys
from collections import Counter

# Standard library modules (Python 3.11+)
STDLIB_MODULES = {
    'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio', 'asyncore',
    'atexit', 'audioop', 'base64', 'bdb', 'binascii', 'binhex', 'bisect', 'builtins',
    'bz2', 'calendar', 'cgi', 'cgitb', 'chunk', 'cmath', 'cmd', 'code', 'codecs',
    'codeop', 'collections', 'colorsys', 'compileall', 'concurrent', 'configparser',
    'contextlib', 'contextvars', 'copy', 'copyreg', 'cProfile', 'crypt', 'csv',
    'ctypes', 'curses', 'dataclasses', 'datetime', 'dbm', 'decimal', 'difflib',
    'dis', 'distutils', 'doctest', 'email', 'encodings', 'enum', 'errno', 'faulthandler',
    'fcntl', 'filecmp', 'fileinput', 'fnmatch', 'formatter', 'fractions', 'ftplib',
    'functools', 'gc', 'getopt', 'getpass', 'gettext', 'glob', 'graphlib', 'grp',
    'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'idlelib', 'imaplib',
    'imghdr', 'imp', 'importlib', 'inspect', 'io', 'ipaddress', 'itertools', 'json',
    'keyword', 'lib2to3', 'linecache', 'locale', 'logging', 'lzma', 'mailbox',
    'mailcap', 'marshal', 'math', 'mimetypes', 'mmap', 'modulefinder', 'multiprocessing',
    'netrc', 'nis', 'nntplib', 'numbers', 'operator', 'optparse', 'os', 'ossaudiodev',
    'parser', 'pathlib', 'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil', 'platform',
    'plistlib', 'poplib', 'posix', 'posixpath', 'pprint', 'profile', 'pstats', 'pty',
    'pwd', 'py_compile', 'pyclbr', 'pydoc', 'queue', 'quopri', 'random', 're',
    'readline', 'reprlib', 'resource', 'rlcompleter', 'runpy', 'sched', 'secrets',
    'select', 'selectors', 'shelve', 'shlex', 'shutil', 'signal', 'site', 'smtpd',
    'smtplib', 'sndhdr', 'socket', 'socketserver', 'spwd', 'sqlite3', 'ssl', 'stat',
    'statistics', 'string', 'stringprep', 'struct', 'subprocess', 'sunau', 'symbol',
    'symtable', 'sys', 'sysconfig', 'syslog', 'tabnanny', 'tarfile', 'telnetlib',
    'tempfile', 'termios', 'test', 'textwrap', 'threading', 'time', 'timeit', 'tkinter',
    'token', 'tokenize', 'tomllib', 'trace', 'traceback', 'tracemalloc', 'tty',
    'turtle', 'turtledemo', 'types', 'typing', 'unicodedata', 'unittest', 'urllib',
    'uu', 'uuid', 'venv', 'warnings', 'wave', 'weakref', 'webbrowser', 'winreg',
    'winsound', 'wsgiref', 'xdrlib', 'xml', 'xmlrpc', 'zipapp', 'zipfile', 'zipimport',
    'zlib', '_thread'
}

def extract_imports_from_code(code):
    """Extract all imports from Python code."""
    imports = set()

    # Match "import module" or "import module as alias"
    import_pattern = r'^\s*import\s+([\w\.]+)(?:\s+as\s+\w+)?'

    # Match "from module import ..." or "from module.submodule import ..."
    from_pattern = r'^\s*from\s+([\w\.]+)\s+import'

    for line in code.split('\n'):
        # Check for regular imports
        match = re.match(import_pattern, line)
        if match:
            module = match.group(1)
            # Get the top-level module name
            top_level = module.split('.')[0]
            imports.add(top_level)

        # Check for from imports
        match = re.match(from_pattern, line)
        if match:
            module = match.group(1)
            # Get the top-level module name
            top_level = module.split('.')[0]
            imports.add(top_level)

    return imports

def extract_python_code_from_trajectory(trajectory_data):
    """Extract all Python code from a trajectory."""
    python_codes = []
    messages = trajectory_data.get('messages', [])

    for i, msg in enumerate(messages):
        if msg['role'] == 'tool' and msg.get('name') == 'python':
            # Get the previous assistant message
            if i > 0 and messages[i-1]['role'] == 'assistant':
                prev_msg = messages[i-1]
                if isinstance(prev_msg['content'], list):
                    for content_item in prev_msg['content']:
                        if isinstance(content_item, dict) and content_item.get('type') == 'text':
                            code = content_item.get('text', '')
                            python_codes.append(code)

    return python_codes

def process_jsonl_file(jsonl_file):
    """Process a single JSONL file and return all imports."""
    all_imports = []

    print(f"Processing: {jsonl_file}")
    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            trajectory = json.loads(line)

            # Extract Python code from this trajectory
            python_codes = extract_python_code_from_trajectory(trajectory)

            # Extract imports from each code snippet
            for code in python_codes:
                imports = extract_imports_from_code(code)
                all_imports.extend(imports)

    print(f"  Found {len(all_imports)} total imports")
    return all_imports

def main(jsonl_files):
    """Main function to extract imports from multiple JSONL files."""
    all_imports = []

    # Process each file
    for jsonl_file in jsonl_files:
        file_imports = process_jsonl_file(jsonl_file)
        all_imports.extend(file_imports)

    print(f"\nTotal imports across all files: {len(all_imports)}")
    print("=" * 60)

    # Filter out standard library modules
    non_stdlib_imports = [imp for imp in all_imports if imp not in STDLIB_MODULES]

    # Count occurrences
    import_counts = Counter(non_stdlib_imports)

    # Print results
    print("\nNon-standard library modules used:")
    print("=" * 60)
    for module, count in import_counts.most_common():
        print(f"{module:30s} : {count:4d}")

    print("\n" + "=" * 60)
    print(f"Total non-stdlib imports: {len(non_stdlib_imports)}")
    print(f"Unique non-stdlib modules: {len(import_counts)}")

    # Print all stdlib modules found (for verification)
    stdlib_imports = [imp for imp in all_imports if imp in STDLIB_MODULES]
    stdlib_counts = Counter(stdlib_imports)
    print(f"\nTotal stdlib imports: {len(stdlib_imports)}")
    print(f"Unique stdlib modules: {len(stdlib_counts)}")

    print("\n\nMost common stdlib modules used:")
    print("=" * 60)
    for module, count in stdlib_counts.most_common(20):
        print(f"{module:30s} : {count:4d}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_imports.py <jsonl_file1> [jsonl_file2] [...]")
        print("\nExample:")
        print("  python extract_imports.py file1.jsonl")
        print("  python extract_imports.py file1.jsonl file2.jsonl file3.jsonl")
        sys.exit(1)

    jsonl_files = sys.argv[1:]
    main(jsonl_files)
