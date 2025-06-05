"""
Core package for the AGI project.

Nothing is imported automatically; sub-modules are loaded explicitly.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0-dev" 