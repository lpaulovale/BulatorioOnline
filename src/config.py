"""
Config compatibility layer.

Re-exports from config.settings for backward compatibility.
"""

from config.settings import Settings, settings, Framework, get_settings

__all__ = ["Settings", "settings", "Framework", "get_settings"]
