"""The per-user cache folder for trained weights and downloaded dictionaries."""

import os
import sys
from pathlib import Path


def cache_dir():
    """Where trained weights and downloads are kept: $LMAZE_CACHE, else the usual per-user cache folder."""
    if os.environ.get("LMAZE_CACHE"):
        return Path(os.environ["LMAZE_CACHE"])
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "lmaze"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "lmaze"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "lmaze"
