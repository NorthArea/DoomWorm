"""Allow ``python -m doomworm``."""

from doomworm.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
