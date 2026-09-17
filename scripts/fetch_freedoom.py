#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Download the Freedoom IWADs, so the worm can play the classic maps.

Freedoom is a free, open replacement for the original Doom data (BSD-style
licence, github.com/freedoom/freedoom). `freedoom1.wad` carries the four
episodes with the E1M1-style map names the classic game uses; ViZDoom ships
only `freedoom2.wad`, which is the Doom II map set.

    uv run scripts/fetch_freedoom.py

The WADs land in `data/wads/` and are git-ignored: they are data, fetched the
same way the connectome is.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data/wads"
API = "https://api.github.com/repos/freedoom/freedoom/releases/latest"
WANTED = ("freedoom1.wad", "freedoom2.wad")


def latest_zip_url() -> str:
    """The `freedoom-<version>.zip` asset of the newest release."""
    with urllib.request.urlopen(API, timeout=30) as response:
        release = json.load(response)
    for asset in release.get("assets", []):
        name = asset.get("name", "")
        if name.startswith("freedoom-") and name.endswith(".zip"):
            return str(asset["browser_download_url"])
    raise SystemExit(f"no freedoom zip in release {release.get('tag_name')!r}")


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    if all((DEST / w).exists() for w in WANTED):
        print(f"already here: {', '.join(str(DEST / w) for w in WANTED)}")
        return 0
    url = latest_zip_url()
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        blob = response.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for member in archive.namelist():
            name = Path(member).name.lower()
            if name in WANTED:
                (DEST / name).write_bytes(archive.read(member))
                print(f"  {name}  {len(archive.read(member)) / 1e6:.1f} MB -> {DEST / name}")
    missing = [w for w in WANTED if not (DEST / w).exists()]
    if missing:
        raise SystemExit(f"the archive did not contain {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
