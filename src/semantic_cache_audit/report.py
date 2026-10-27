"""Rendering: the plain text tables the CLI prints and the HTML report evidence frames.

Kept out of the audit package on purpose. Nothing in `audit/` should know how a number
is going to be displayed, because the moment it does, a presentation change can move a
published figure.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence

from .audit.budget import Decision, Exit
from .audit.sweep import Point, Sweep

VERDICT_TEXT = {
    Exit.OK: "THRESHOLD FOUND",
    Exit.NO_THRESHOLD: "NO THRESHOLD MEETS THE BUDGET",
    Exit.REFUSED: "REFUSED, THE CORPUS CANNOT RESOLVE THIS BUDGET",
    Exit.ERROR: "TOOL ERROR",
}


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A fixed width table. Used for stdout, never for a committed document."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join("-" * w for w in widths)
    out = ["  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)), line]
    out += ["  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) for row in rows]
    return "\n".join(out)


def pct(value: float, places: int = 2) -> str:
    return f"{value * 100:.{places}f}%"


def decision_text(decision: Decision, sweep: Sweep) -> str:
    """The block the CLI prints. The uncomfortable line comes first."""
    lines = [
        f"embedder                  {sweep.embedder}",
        f"separation                {sweep.separation_auc:.3f}   (0.5 is chance; below "
        "0.5 means near misses outrank true paraphrases)",
        f"most confident wrong run  {sweep.wrong_prefix_hits} hits before the first "
        "correct one",
        f"servable with no wrong hit {decision.clean_ceiling_hits} of "
        f"{decision.positive_controls} true paraphrases",
        f"corpus                    {len(sweep.pairs)} pairs, "
        f"{decision.positive_controls} positive controls",
        f"tightest budget provable  {pct(decision.tightest_certifiable_budget)} "
        f"at {pct(decision.confidence, 0)} confidence",
        "",
        f"verdict                   {VERDICT_TEXT[decision.exit_code]}  "
        f"(exit {int(decision.exit_code)})",
        f"reason                    {decision.reason}",
    ]
    if decision.operating_point is not None:
        point = decision.operating_point
        lines += [
            "",
            f"threshold                 {point.threshold:.4f}",
            f"hit rate                  {pct(point.hit_rate)} of probes "
            f"({point.hits} of {point.probes})",
            f"false hit rate            {pct(point.false_hit_rate)} of hits, bounded at "
            f"{pct(point.false_hit_rate_upper)}",
            f"at declared prevalence    {pct(point.weighted_false_hit_rate)} of hits, "
            f"hit rate {pct(point.weighted_hit_rate)}",
            f"resolution floor          below one in {point.hits}" if point.hits else "",
            f"missed paraphrases        {point.missed_opportunities}",
            f"worst near miss class     {decision.worst_class} at "
            f"{pct(decision.worst_class_false_hit_rate)}",
        ]
    for note in decision.notes:
        lines.append(f"note                      {note}")
    return "\n".join(lines)


def class_table(point: Point) -> str:
    rows = [
        [
            cls,
            str(stats["probes"]),
            str(stats["hits"]),
            str(stats["false_hits"]),
            pct(stats["wrongly_served_rate"]),
            pct(stats["wrongly_served_upper"]),
        ]
        for cls, stats in sorted(
            point.by_class.items(), key=lambda kv: (-kv[1]["wrongly_served_rate"], kv[0])
        )
    ]
    return table(
        ["class", "probes", "hits", "wrong", "wrongly served", "upper bound"], rows
    )


def curve_table(sweep: Sweep, every: int = 10) -> str:
    rows = [
        [
            f"{p.threshold:.4f}",
            pct(p.hit_rate, 1),
            pct(p.false_hit_rate, 1),
            pct(p.false_hit_rate_upper, 1),
            str(p.hits),
            str(p.false_hits),
            str(p.missed_opportunities),
        ]
        for p in sweep.points[::every]
    ]
    return table(
        ["threshold", "hit rate", "wrong share", "upper", "hits", "wrong", "missed"], rows
    )


def comparison_table(results: list[tuple[Sweep, Decision]]) -> str:
    rows = []
    for sweep, decision in results:
        point = decision.operating_point
        rows.append(
            [
                sweep.embedder,
                f"{sweep.separation_auc:.3f}",
                str(sweep.wrong_prefix_hits),
                decision.exit_code.name,
                f"{point.threshold:.4f}" if point else "-",
                pct(point.hit_rate, 1) if point else "-",
                pct(point.false_hit_rate, 1) if point else "-",
            ]
        )
    return table(
        ["embedder", "separation", "wrong run", "verdict", "threshold", "hit rate", "wrong share"],
        rows,
    )


# ------------------------------------------------------------------- HTML report


_CSS = """
:root { --ink:#15202b; --muted:#5b6b7a; --rule:#d8dee4; --bg:#ffffff; --panel:#f6f8fa; }
* { box-sizing: border-box; }
body { margin:0; padding:40px; background:var(--bg); color:var(--ink);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-0.01em; }
h2 { font-size:19px; margin:38px 0 12px; padding-bottom:7px; border-bottom:2px solid var(--rule); }
p.sub { color:var(--muted); margin:0 0 26px; }
table { border-collapse:collapse; width:100%; margin:10px 0 4px; font-size:14px; }
th,td { text-align:left; padding:7px 11px; border-bottom:1px solid var(--rule); }
th { background:var(--panel); font-weight:600; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
.verdict { display:inline-block; padding:7px 15px; border-radius:4px; font-weight:700;
  font-size:14px; letter-spacing:0.02em; }
.verdict.ok { background:#1a7f37; color:#ffffff; }
.verdict.no { background:#9a3412; color:#ffffff; }
.verdict.refused { background:#7a4f01; color:#ffffff; }
.kv { display:grid; grid-template-columns:260px 1fr; gap:5px 18px; margin:14px 0; font-size:14px; }
.kv dt { color:var(--muted); }
.kv dd { margin:0; font-variant-numeric:tabular-nums; }
.note { background:var(--panel); border-left:3px solid var(--muted); padding:11px 15px;
  margin:12px 0; font-size:14px; color:var(--ink); }
.bar { height:11px; background:#e6eaef; border-radius:2px; overflow:hidden; min-width:90px; }
.bar > span { display:block; height:100%; background:#9a3412; }
footer { margin-top:44px; color:var(--muted); font-size:13px; border-top:1px solid var(--rule);
  padding-top:14px; }
"""

_VERDICT_CLASS = {Exit.OK: "ok", Exit.NO_THRESHOLD: "no", Exit.REFUSED: "refused", Exit.ERROR: "no"}


def _rows_html(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f'<th class="num">{html.escape(h)}</th>' if i else f"<th>{html.escape(h)}</th>"
                   for i, h in enumerate(headers))
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="num">{c}</td>' if i else f"<td>{c}</td>" for i, c in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def html_report(sweep: Sweep, decision: Decision, metrics: dict | None = None) -> str:
    """The report `tools/capture_screenshots.py` frames by heading.

    Headings are stable identifiers, not decoration: the screenshot tool scrolls to a
    named h2 and clips to the next one, and fails the run when a heading disappears.
    """
    point = decision.operating_point
    verdict_cls = _VERDICT_CLASS[decision.exit_code]
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Semantic cache audit</title>",
        f"<style>{_CSS}</style></head><body>",
        "<h1>Semantic cache audit</h1>",
        f"<p class='sub'>embedder <strong>{html.escape(sweep.embedder)}</strong>, "
        f"{len(sweep.pairs)} labelled pairs, "
        f"{decision.positive_controls} positive controls</p>",
        "<h2 id='verdict'>Verdict</h2>",
        f"<p><span class='verdict {verdict_cls}'>{VERDICT_TEXT[decision.exit_code]}</span>"
        f" &nbsp; exit code {int(decision.exit_code)}</p>",
        f"<div class='note'>{html.escape(decision.reason)}</div>",
        "<dl class='kv'>",
        f"<dt>separation</dt><dd>{sweep.separation_auc:.3f}</dd>",
        f"<dt>most confident wrong run</dt><dd>{sweep.wrong_prefix_hits} hits</dd>",
        f"<dt>servable with no wrong hit</dt><dd>{decision.clean_ceiling_hits} of "
        f"{decision.positive_controls} true paraphrases</dd>",
        f"<dt>tightest budget provable</dt>"
        f"<dd>{pct(decision.tightest_certifiable_budget)}</dd>",
        f"<dt>requested budget</dt><dd>{pct(decision.budget)}</dd>",
        "</dl>",
    ]
    if point is not None:
        parts += [
            "<h2 id='operating-point'>Operating point</h2>",
            _rows_html(
                ["measure", "value"],
                [
                    ["threshold", f"{point.threshold:.4f}"],
                    ["hit rate", f"{pct(point.hit_rate)} ({point.hits} of {point.probes})"],
                    ["false hit rate", pct(point.false_hit_rate)],
                    ["upper bound on false hit rate", pct(point.false_hit_rate_upper)],
                    ["at declared prevalence", pct(point.weighted_false_hit_rate)],
                    ["resolution floor", f"below one in {point.hits}" if point.hits else "n/a"],
                    ["missed paraphrases", str(point.missed_opportunities)],
                ],
            ),
            "<h2 id='by-class'>Where it goes wrong</h2>",
            _rows_html(
                ["class", "probes", "hits", "wrong", "wrongly served", ""],
                [
                    [
                        cls,
                        str(s["probes"]),
                        str(s["hits"]),
                        str(s["false_hits"]),
                        pct(s["wrongly_served_rate"]),
                        f"<div class='bar'><span style='width:"
                        f"{min(s['wrongly_served_rate'] * 100, 100):.0f}%'></span></div>",
                    ]
                    for cls, s in sorted(
                        point.by_class.items(),
                        key=lambda kv: (-kv[1]["wrongly_served_rate"], kv[0]),
                    )
                ],
            ),
        ]
    parts += [
        "<h2 id='curve'>What the threshold buys and what it costs</h2>",
        _rows_html(
            ["threshold", "hit rate", "wrong share", "upper bound", "missed paraphrases"],
            [
                [
                    f"{p.threshold:.4f}",
                    pct(p.hit_rate, 1),
                    pct(p.false_hit_rate, 1),
                    pct(p.false_hit_rate_upper, 1),
                    str(p.missed_opportunities),
                ]
                for p in sweep.points[::10]
            ],
        ),
        "<h2 id='notes'>What this report does not establish</h2>",
        "<div class='note'>Every figure above is a fact about one embedder over one "
        "corpus. The corpus is deliberately near miss heavy, so the unweighted rates "
        "describe an adversarial stream rather than production traffic; the prevalence "
        "weighted rate is the one to quote, and the prevalence it uses is declared "
        "rather than observed.</div>",
        "<footer>Generated by semantic-cache-audit. Every number here is produced by "
        "<code>tools/collect_metrics.py</code> and checked against the documents by "
        "<code>tools/check_numbers.py</code>."
        + (f" metrics.json carries {len(metrics.get('metrics', {}))} values." if metrics else "")
        + "</footer></body></html>",
    ]
    return "".join(parts)


def as_json(sweep: Sweep, decision: Decision) -> str:
    return json.dumps({"sweep": sweep.to_dict(), "decision": decision.to_dict()}, indent=2)
