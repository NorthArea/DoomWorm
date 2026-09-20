"""Which published brains is anything still pointing at, and do they still load.

    uv run python scripts/audit_brains.py            # the report
    uv run python scripts/audit_brains.py --verify   # also load every kept brain

The repository's rule is that a published row replays from what is committed, so
a brain file is evidence until something says otherwise. This asks, for every
one of them, whether *anything* still refers to it -- by path, by the directory
it sits in, or by the name of the lane that produced it, because a findings row
citing a lane is citing all of it.

Nothing here deletes. It prints what could go and what it weighs, and the
decision stays with a person.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRAIN_ROOTS = (ROOT / "docs",)
TEXT = (".md", ".py", ".sh", ".toml", ".cfg", ".yml", ".yaml")


def everything_written() -> str:
    """Every word this repository writes about itself, concatenated."""
    out = [(ROOT / "Makefile").read_text(errors="ignore")]
    for folder in ("docs", "scripts", "tests", "src"):
        for f in (ROOT / folder).rglob("*"):
            if f.is_file() and f.suffix in TEXT:
                out.append(f.read_text(errors="ignore"))
    return "\n".join(out)


def lane_of(brain: Path) -> Path:
    """The lane a brain belongs to: the directory under `brains/`."""
    parts = brain.relative_to(ROOT).parts
    i = parts.index("brains")
    return Path(*parts[: i + 2]) if len(parts) > i + 1 else Path(*parts[: i + 1])


def audit() -> tuple[dict[Path, list[Path]], dict[Path, list[Path]]]:
    """Group the brains by lane, into the lanes something cites and the rest."""
    text = everything_written()
    brains = sorted(
        p for root in BRAIN_ROOTS for p in root.rglob("*.json")
        if "brains" in p.parts and not p.name.endswith(".meta.json")
    )  # fmt: skip
    kept: dict[Path, list[Path]] = defaultdict(list)
    loose: dict[Path, list[Path]] = defaultdict(list)
    for b in brains:
        rel = b.relative_to(ROOT).as_posix()
        lane = lane_of(b)
        cited = (
            rel in text
            or b.parent.relative_to(ROOT).as_posix() in text
            or lane.as_posix() in text
            or lane.name in text
        )
        (kept if cited else loose)[lane].append(b)
    return kept, loose


def weight(brains: list[Path]) -> float:
    """Megabytes of the brains and everything that travels with them.

    Counted through a set, because a brain, its `.meta.json` and its `.csv` all
    match more than one of the patterns below and the first version of this
    added several of them twice -- which is how a 80 MB tree reported 140.
    """
    files = set()
    for b in brains:
        files.add(b)
        files.update(extra for extra in b.parent.glob(f"{b.stem}*") if extra.is_file())
    return sum(f.stat().st_size for f in files) / 1e6


def verify(brains: list[Path]) -> list[str]:
    """Actually build each brain, which is the only check that means anything.

    Reading the JSON and looking for a key proves nothing -- the first version of
    this did that, guessed the wrong key, and reported every brain in the
    repository as broken.
    """
    import broomworm  # noqa: F401  -- registers the robot track's sensor presets
    from wormlab.candidates import load_candidate

    bad = []
    for b in brains:
        # a brain was trained behind some sensor preset and the file does not
        # always say which, so try them; and not every candidate exposes
        # weights -- PPO keeps its policy elsewhere -- so loading is the check
        for sensors in ("ideal", "vacuum", "car", "noisy"):
            try:
                load_candidate(b, maps="doom4", task="doom", sensors=sensors)
                break
            except ValueError:
                continue
            except Exception as exc:
                bad.append(f"{b.relative_to(ROOT)}: {type(exc).__name__}: {exc}")
                break
        else:
            bad.append(f"{b.relative_to(ROOT)}: no sensor preset loads it")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true", help="load every kept brain")
    args = ap.parse_args()
    kept, loose = audit()

    print("lanes something still points at\n")
    for lane in sorted(kept, key=lambda p: -weight(kept[p])):
        print(f"  {weight(kept[lane]):6.1f} MB  {len(kept[lane]):3d} brains  {lane}")
    print(f"\n  kept in total: {sum(weight(v) for v in kept.values()):.1f} MB\n")

    print("lanes nothing points at -- by path, by directory, or by name\n")
    for lane in sorted(loose, key=lambda p: -weight(loose[p])):
        print(f"  {weight(loose[lane]):6.1f} MB  {len(loose[lane]):3d} brains  {lane}")
    print(f"\n  could go: {sum(weight(v) for v in loose.values()):.1f} MB")

    if args.verify:
        flat = [b for v in kept.values() for b in v]
        print(f"\nloading all {len(flat)} kept brains...")
        bad = verify(flat)
        print("  every one of them loads" if not bad else "  BROKEN:\n    " + "\n    ".join(bad))
    return 0


if __name__ == "__main__":
    sys.exit(main())
