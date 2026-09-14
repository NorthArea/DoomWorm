"""Command-line entry point.

The final pipeline is driven by ``doomworm train`` and ``doomworm play``.
Subcommands are registered here as they become available stage by stage.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from doomworm import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="doomworm", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_subparsers(dest="command", title="commands")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return 0  # pragma: no cover - no subcommands registered yet
