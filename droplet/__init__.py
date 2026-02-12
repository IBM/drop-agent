"""DROP: Deep Research On Premise Agent"""

try:
    from importlib.metadata import version
    __version__ = version("droplet")
except Exception:
    __version__ = "dev"
