"""Regenerates the committed embedding matrices.

Two modes, and the split is the offline rule made operational. Without `--pretrained`
this writes only the fixtures that need nothing but this repository, so it runs on a
machine with no network. With `--pretrained` it also fetches the encoder weights and
recomputes the matrix every headline figure is read from, which is the only command
in the repository that reaches outside it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_cache_audit.corpus.loader import load_pairs  # noqa: E402
from semantic_cache_audit.embed.fixtures import write_fixture  # noqa: E402
from semantic_cache_audit.embed.registry import (  # noqa: E402
    PRETRAINED_NAME,
    offline_fixture_names,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="also recompute the pretrained matrix. Needs the encoder extra and network.",
    )
    args = parser.parse_args(argv)
    pairs = load_pairs()
    names = list(offline_fixture_names())
    if args.pretrained:
        names.insert(0, PRETRAINED_NAME)
    for name in names:
        record = write_fixture(name, pairs)
        print(f"{record['embedder']:44s} dim={record['dim']:5d} rows={record['rows']}")
    if not args.pretrained:
        print(
            f"\nleft {PRETRAINED_NAME} alone. Run with --pretrained to recompute it, "
            "which needs the encoder extra and a reachable model host."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
