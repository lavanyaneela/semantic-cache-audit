"""Checks that regenerating the embedding fixtures reproduces the geometry they encode.

Not their bytes. The distinction is the point, and it cost a red build to learn.

`fixtures/embeddings/lsa.npz` is produced by a truncated SVD, and LAPACK is free to
return any orthonormal basis for a degenerate singular subspace. Which one it returns
depends on the BLAS kernel, which depends on the core count and the CPU. Measured on
one machine by varying the thread count alone: individual coordinates move by up to
1.061, while every pairwise cosine moves by at most 2.4e-07. The coordinates are not
identifiable. The geometry is.

That matters because the audit never reads a coordinate. Every published number is a
function of inner products between corpus vectors: the threshold sweep compares pair
similarities, the separation statistic ranks them, the budget decision counts how many
fall above a threshold. So this check regenerates each offline fixture and compares the
full matrix of pairwise cosines against the committed one, which is the strongest claim
the mathematics actually supports.

A byte comparison would be a stronger claim and a false one. It passed on the machine
that built the fixtures and failed on a runner with a different core count, which is
the worst of both: it does not hold, and it fails for a reason unrelated to anything
being wrong.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_cache_audit.corpus.loader import load_pairs  # noqa: E402
from semantic_cache_audit.embed.fixtures import load_fixture, write_fixture  # noqa: E402
from semantic_cache_audit.embed.registry import offline_fixture_names  # noqa: E402

#: Chosen two orders of magnitude above the largest deviation observed from varying
#: the BLAS thread count, and far below the precision of any published figure: the
#: tightest rendered value is a cosine at three decimal places.
COSINE_TOLERANCE = 1e-5


def gram(matrix: np.ndarray) -> np.ndarray:
    """Every pairwise cosine. The rows are unit vectors, so this is the inner product."""
    return matrix @ matrix.T


def check(name: str, pairs, scratch: Path) -> tuple[bool, float, float]:
    """Regenerate one fixture and return whether it matches, and by how much it moved."""
    committed_texts, committed = load_fixture(name)
    write_fixture(name, pairs, scratch)
    regenerated_texts, regenerated = load_fixture(name, scratch, verify=False)
    if committed_texts != regenerated_texts:
        raise SystemExit(
            f"{name}: the regenerated fixture covers different queries than the "
            "committed one, so the corpus and the fixture are out of step"
        )
    coordinate_shift = float(np.abs(committed - regenerated).max())
    cosine_shift = float(np.abs(gram(committed) - gram(regenerated)).max())
    return cosine_shift <= COSINE_TOLERANCE, coordinate_shift, cosine_shift


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tolerance", type=float, default=COSINE_TOLERANCE,
        help="maximum permitted movement in any pairwise cosine",
    )
    args = parser.parse_args(argv)
    pairs = load_pairs()
    failures: list[str] = []
    print(f"{'fixture':44s} {'coordinates':>13s} {'cosines':>12s}")
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        for name in offline_fixture_names():
            _, coordinate_shift, cosine_shift = check(name, pairs, scratch)
            ok = cosine_shift <= args.tolerance
            flag = "" if ok else "   REPRODUCIBILITY FAILURE"
            print(f"{name:44s} {coordinate_shift:13.3e} {cosine_shift:12.3e}{flag}")
            if not ok:
                failures.append(name)
    if failures:
        print(
            f"\n{len(failures)} fixture(s) did not reproduce their geometry within "
            f"{args.tolerance:g}: {failures}",
            file=sys.stderr,
        )
        return 1
    print(
        f"\nevery offline fixture reproduces every pairwise cosine within "
        f"{args.tolerance:g}. Coordinates are allowed to move, because a truncated "
        "SVD does not define them uniquely and nothing here reads one."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
