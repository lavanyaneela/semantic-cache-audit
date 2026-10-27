"""The command line surface, and the exit codes that make it usable from a pipeline.

The codes matter more than the output. A tool that exits zero whenever it ran tells a
pipeline nothing; this one distinguishes "a threshold meets your budget" from "none
does" from "your corpus cannot answer that question", and a build can branch on all
three.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import report
from .audit.budget import Exit, decide
from .audit.sweep import run_sweep
from .config import DEFAULTS
from .corpus.loader import CorpusError, load_manifest, load_pairs
from .embed.fixtures import FixtureError, available, load_fixture
from .embed.registry import UnknownEmbedder, fixture_names, validate_name


def _load(embedder: str):
    validate_name(embedder)
    pairs = load_pairs()
    texts, matrix = load_fixture(embedder)
    return pairs, texts, matrix


def _sweep_and_decide(args):
    pairs, texts, matrix = _load(args.embedder)
    sweep = run_sweep(
        pairs,
        texts,
        matrix,
        args.embedder,
        steps=args.steps,
        prevalence=args.prevalence,
    )
    decision = decide(sweep, budget=args.budget, min_hit_rate=args.min_hit_rate)
    return sweep, decision


def cmd_audit(args) -> int:
    sweep, decision = _sweep_and_decide(args)
    if args.json:
        print(report.as_json(sweep, decision))
    else:
        print(report.decision_text(decision, sweep))
        if decision.operating_point is not None:
            print()
            print(report.class_table(decision.operating_point))
    return int(decision.exit_code)


def cmd_sweep(args) -> int:
    sweep, decision = _sweep_and_decide(args)
    print(report.curve_table(sweep, every=args.every))
    return int(decision.exit_code)


def cmd_classes(args) -> int:
    """Per class breakdown, by default at the reference threshold rather than the
    budget operating point.

    On an inverted embedder the budget operating point is always the fully permissive
    threshold, where every class sits at one hundred percent and the table says
    nothing. The reference threshold, the median positive control score, is where E2
    measures: half the true paraphrases are served, so each class has its full set of
    probes behind its number. `--at operating-point` asks for the other one.
    """
    import numpy as np

    from .audit.probe import pair_scores, probe_at_threshold
    from .audit.sweep import summarise

    sweep, decision = _sweep_and_decide(args)
    if args.at == "operating-point":
        point = decision.operating_point
        if point is None:
            print("no usable threshold, so there is no per class breakdown to show")
            return int(decision.exit_code)
        label = f"budget operating point, threshold {point.threshold:.4f}"
    else:
        pairs = sweep.pairs
        scores = pair_scores(pairs, *_load(args.embedder)[1:])
        threshold = float(np.median([s for p, s in zip(pairs, scores, strict=True)
                                     if p.same_answer]))
        point = summarise(
            probe_at_threshold(pairs, scores, threshold), threshold, pairs, args.prevalence
        )
        label = (
            f"reference threshold {threshold:.4f}, where half the true paraphrases hit"
        )
    print(label)
    print()
    print(report.class_table(point))
    return int(decision.exit_code)


def cmd_compare(args) -> int:
    names = args.embedders or [n for n in fixture_names() if n in set(available())]
    results = []
    pairs = load_pairs()
    for name in names:
        validate_name(name)
        texts, matrix = load_fixture(name)
        sweep = run_sweep(pairs, texts, matrix, name, steps=args.steps, prevalence=args.prevalence)
        results.append((sweep, decide(sweep, budget=args.budget, min_hit_rate=args.min_hit_rate)))
    print(report.comparison_table(results))
    # The comparison is informational. It exits zero when it ran, because "which of
    # these is best" is not a pass or fail question and a pipeline branching on it
    # would be branching on the wrong thing.
    return int(Exit.OK)


def cmd_corpus(args) -> int:
    manifest = load_manifest()
    if args.json:
        print(json.dumps(manifest, indent=2))
        return int(Exit.OK)
    rows = [[k, str(v)] for k, v in manifest.items() if not isinstance(v, dict)]
    print(report.table(["field", "value"], rows))
    print()
    print(report.table(["class", "pairs"], [[k, str(v)] for k, v in manifest["by_class"].items()]))
    return int(Exit.OK)


def cmd_report(args) -> int:
    sweep, decision = _sweep_and_decide(args)
    metrics_path = Path("docs/metrics.json")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.html_report(sweep, decision, metrics), encoding="utf-8")
    print(f"wrote {out}")
    return int(decision.exit_code)


def cmd_serve(args) -> int:  # pragma: no cover - a server is not run by the suite
    import uvicorn

    from .api.app import build_app

    uvicorn.run(build_app(args.embedder, threshold=args.threshold), host=args.host, port=args.port)
    return int(Exit.OK)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-cache-audit",
        description=(
            "Measures what a semantic cache gets wrong, not just how often it hits. "
            "Exit 0 a threshold meets the budget, 1 none does, 2 the corpus cannot "
            "resolve the budget, 3 tool error."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(p, with_budget=True):
        p.add_argument("--embedder", default=DEFAULTS.embedder, help="embedder fixture to audit")
        p.add_argument("--steps", type=int, default=DEFAULTS.threshold_steps)
        p.add_argument("--prevalence", type=float, default=DEFAULTS.near_miss_prevalence)
        if with_budget:
            p.add_argument("--budget", type=float, default=DEFAULTS.budget)
            p.add_argument("--min-hit-rate", type=float, default=DEFAULTS.min_hit_rate)
        return p

    a = shared(sub.add_parser("audit", help="decide whether any threshold meets the budget"))
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_audit)

    s = shared(sub.add_parser("sweep", help="print the hit rate against false hit rate curve"))
    s.add_argument("--every", type=int, default=10, help="print every Nth grid point")
    s.set_defaults(func=cmd_sweep)

    c = shared(sub.add_parser("classes", help="false hit rate by adversarial class"))
    c.add_argument(
        "--at",
        choices=("reference", "operating-point"),
        default="reference",
        help="reference is the median positive control score, where E2 measures",
    )
    c.set_defaults(func=cmd_classes)

    m = shared(sub.add_parser("compare", help="every embedder fixture side by side"))
    m.add_argument("--embedders", nargs="*", default=None)
    m.set_defaults(func=cmd_compare)

    p = sub.add_parser("corpus", help="corpus composition and provenance")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_corpus)

    r = shared(sub.add_parser("report", help="write the HTML report"))
    r.add_argument("--out", default="docs/evidence/report-budget-005.html")
    r.set_defaults(func=cmd_report)

    v = sub.add_parser("serve", help="run the REST surface (not part of make verify)")
    v.add_argument("--embedder", default=DEFAULTS.embedder)
    v.add_argument("--threshold", type=float, default=0.9)
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)
    v.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate(args)
        return int(args.func(args))
    except (CorpusError, FixtureError, UnknownEmbedder) as exc:
        # The three ways the tool can be unable to start. Exit 3, because the tool
        # failed rather than answered, and a pipeline must be able to tell that apart
        # from a cache that should not ship.
        print(f"error: {exc}", file=sys.stderr)
        return int(Exit.ERROR)
    except Exception as exc:  # noqa: BLE001
        # Everything else also exits 3. It used to escape as a traceback, which the
        # shell reports as 1, so a crash was indistinguishable from "this cache
        # should not ship" to anything branching on the code. The cost of catching
        # broadly is that a genuine bug now looks like a user error, so SCA_DEBUG=1
        # re-raises and gives back the traceback.
        if os.environ.get("SCA_DEBUG"):
            raise
        print(
            f"error: {type(exc).__name__}: {exc}\n"
            "set SCA_DEBUG=1 to see the traceback if this looks like a bug",
            file=sys.stderr,
        )
        return int(Exit.ERROR)


def _validate(args) -> None:
    """Refuse nonsense at the boundary rather than producing a nonsense number.

    `--budget` was validated and `--prevalence` was not, so a prevalence of 5 printed
    a weighted rate of 500 percent and a negative one printed a negative hit rate.
    """
    steps = getattr(args, "steps", None)
    if steps is not None and steps < 1:
        raise ValueError(f"--steps must be at least 1, got {steps}")
    prevalence = getattr(args, "prevalence", None)
    if prevalence is not None and not 0.0 <= prevalence <= 1.0:
        raise ValueError(f"--prevalence is a share and must be in [0, 1], got {prevalence}")
    min_hit_rate = getattr(args, "min_hit_rate", None)
    if min_hit_rate is not None and not 0.0 <= min_hit_rate <= 1.0:
        raise ValueError(f"--min-hit-rate is a share and must be in [0, 1], got {min_hit_rate}")


if __name__ == "__main__":  # pragma: no cover - thin entry point
    raise SystemExit(main())
