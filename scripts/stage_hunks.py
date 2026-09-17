#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Stage selected hunks of a file, so one commit carries one stage.

Track B was built in one working tree: `world.py` holds the mini-Doom gun (B1),
the angled walls of a stock map (B11) and the prey channels (B2d), and the same
is true of the mappings, the CLI and the Makefile. `git add -p` is interactive
and not available here, so this picks hunks by what they contain:

    uv run scripts/stage_hunks.py src/.../world.py --with prey_ segments
    uv run scripts/stage_hunks.py src/.../world.py --without prey_ segments

`--with` stages every hunk containing any of the words, `--without` stages
every hunk containing none of them; both read the *unstaged* diff, so they can
be run one after the other for consecutive commits.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def diff(path: str) -> str:
    """Unstaged diff of one file, with enough context to apply cleanly."""
    out = subprocess.run(
        ["git", "diff", "-U3", "--", path], capture_output=True, text=True, check=True
    )
    return out.stdout


def split(text: str) -> tuple[list[str], list[str]]:
    """Header lines and the hunks that follow them."""
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.startswith("@@")), len(lines))
    header, hunks, current = lines[:start], [], []
    for line in lines[start:]:
        if line.startswith("@@") and current:
            hunks.append("".join(current))
            current = []
        current.append(line)
    if current:
        hunks.append("".join(current))
    return header, hunks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--with", dest="keep", nargs="*", default=[], help="stage hunks containing")
    ap.add_argument("--without", dest="drop", nargs="*", default=[], help="stage hunks missing all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    header, hunks = split(diff(args.path))
    if not hunks:
        print(f"{args.path}: nothing unstaged")
        return 0
    if args.keep:
        chosen = [h for h in hunks if any(word in h for word in args.keep)]
    elif args.drop:
        chosen = [h for h in hunks if not any(word in h for word in args.drop)]
    else:
        chosen = hunks
    print(f"{args.path}: {len(chosen)} of {len(hunks)} hunks")
    if args.dry_run or not chosen:
        for h in chosen:
            print("   ", h.splitlines()[0])
        return 0

    patch = "".join(header) + "".join(chosen)
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as fh:
        fh.write(patch)
        name = fh.name
    result = subprocess.run(["git", "apply", "--cached", name], capture_output=True, text=True)
    Path(name).unlink()
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
