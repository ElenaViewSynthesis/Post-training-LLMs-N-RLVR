"""Retired pilot for the superseded 225K oversampling pipeline."""

from __future__ import annotations

import sys


MIGRATION_MESSAGE = (
    "This oversampling pilot is retired and cannot generate provider requests. "
    "Use refinement_pilot.py for an isolated streaming-pipeline pilot."
)


def main() -> int:
    print(MIGRATION_MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
