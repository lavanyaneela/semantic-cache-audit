"""Captures framed screenshots of the report the tool actually produces.

Three rules, each from a failure.

**Frame by heading, not by pixel offset.** Scroll to a named heading, read its
document offset, and clip to the next named heading. A hard coded crop drifts
silently the first time a paragraph gets longer, and then the README shows half a
table. A missing heading fails the run, because it means the report changed shape and
the shot would be meaningless.

**`full_page=True` even when a clip is given.** Without it Chromium clamps the clip to
the viewport and every shot comes out exactly one viewport tall with the bottom
missing.

**Read back the computed style of the verdict badge and fail if it is invisible.** A
badge class and a table cell class can collide at equal specificity, the later rule
wins, and the verdict renders unreadable. The Python is correct, the HTML is correct,
the tests pass, and the headline is invisible. A screenshot tool is the only thing in
the pipeline that can see pixels, so the check belongs here.

The report is built rather than photographed from disk, because a screenshot of a
stale report is worse than no screenshot.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "evidence"
EMBEDDER = "st:sentence-transformers/all-MiniLM-L6-v2"

#: name -> (budget, heading to start at, heading to stop before). The second heading
#: may be None to run to the end of the document.
#:
#: Two budgets, because the report genuinely has two shapes. At a budget the corpus
#: cannot resolve there is no operating point, so the per class and curve sections do
#: not exist, and the first version of this tool failed here rather than silently
#: photographing the wrong part of the page. That failure is the behaviour wanted.
SHOTS: dict[str, tuple[str, str, str | None]] = {
    "verdict": ("0.001", "verdict", "curve"),
    "by-class": ("0.05", "by-class", "curve"),
    "curve": ("0.05", "curve", "notes"),
}


def build_report(budget: str, path: Path) -> None:
    subprocess.run(
        [
            sys.executable, "-m", "semantic_cache_audit.cli", "report",
            "--embedder", EMBEDDER, "--budget", budget, "--out", str(path),
        ],
        cwd=ROOT,
        check=False,
    )
    if not path.exists():
        raise SystemExit("the report was not written, so there is nothing to photograph")


def capture() -> dict:
    from playwright.sync_api import sync_playwright

    manifest: dict = {"embedder": EMBEDDER, "shots": []}
    budgets = sorted({budget for budget, _, _ in SHOTS.values()})
    reports = {}
    for budget in budgets:
        path = OUT / f"report-budget-{budget.replace('.', '')}.html"
        build_report(budget, path)
        reports[budget] = path
    with sync_playwright() as play:
        browser = play.chromium.launch()
        page = browser.new_page(viewport={"width": 1100, "height": 900},
                                device_scale_factor=2)
        for budget, path in reports.items():
            page.goto(path.resolve().as_uri())
            page.wait_for_timeout(250)

            badge = page.evaluate(
                """() => {
                    const el = document.querySelector('.verdict');
                    if (!el) return null;
                    const s = getComputedStyle(el);
                    return {
                        text: el.textContent.trim(),
                        colour: s.color,
                        background: s.backgroundColor,
                        visible: el.offsetWidth > 0 && el.offsetHeight > 0,
                    };
                }"""
            )
            if badge is None or not badge["visible"]:
                raise SystemExit(
                    f"the verdict badge is missing or has no size at budget {budget}"
                )
            if badge["colour"] == badge["background"]:
                raise SystemExit(
                    f"at budget {budget} the verdict badge renders {badge['colour']} on "
                    f"{badge['background']}, so the headline is invisible. A CSS rule "
                    "is colliding."
                )
            manifest.setdefault("badges", {})[budget] = badge

            for name, (shot_budget, start, stop) in SHOTS.items():
                if shot_budget != budget:
                    continue
                box = page.evaluate(
                    """([start, stop]) => {
                        const a = document.getElementById(start);
                        if (!a) return null;
                        const b = stop ? document.getElementById(stop) : null;
                        const top = a.getBoundingClientRect().top + window.scrollY - 16;
                        const bottom = b
                            ? b.getBoundingClientRect().top + window.scrollY - 16
                            : document.body.scrollHeight;
                        return {top, height: Math.max(bottom - top, 80)};
                    }""",
                    [start, stop],
                )
                if box is None:
                    raise SystemExit(
                        f"the report at budget {budget} has no heading with id {start!r}. "
                        "It changed shape, so this screenshot would no longer show what "
                        "its caption says."
                    )
                image = OUT / f"{name}.png"
                page.screenshot(
                    path=str(image),
                    full_page=True,
                    clip={"x": 0, "y": box["top"], "width": 1100, "height": box["height"]},
                )
                manifest["shots"].append(
                    {
                        "file": image.name,
                        "report": path.name,
                        "budget": budget,
                        "framed_by": [start, stop],
                        "height": round(box["height"]),
                    }
                )
        browser.close()
    return manifest


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = capture()
    (OUT / "screenshot-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"captured {len(manifest['shots'])} framed shots")
    for budget, badge in manifest["badges"].items():
        print(f"  budget {budget}: badge reads {badge['text']!r} in "
              f"{badge['colour']} on {badge['background']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
