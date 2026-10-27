"""Renders the README's diagram source to a committed SVG.

GitHub's own Mermaid renderer fails intermittently with a message that comes from
inside its lazily loaded bundle, so there is nothing to fix in the source and no way
to fix it from the repository. Rendering here and committing the result means the
diagram displays identically on GitHub, in an editor preview, in a PDF export and
offline.

Four details, each of which produced a broken image before it was fixed:

* `htmlLabels` must be off. With it on, Mermaid puts HTML inside a `foreignObject`,
  including unclosed `br` tags, so the file is not well formed XML. It still displays
  when injected into a live page and fails to load as an `img src`, with a broken
  image icon and nothing in any console to explain it.
* The root element needs intrinsic dimensions. Mermaid emits a percentage width and
  no height, which is right for a live page and leaves a browser with no aspect ratio
  for an `img src`.
* The background must be painted. Transparent sounds theme neutral and is not: node
  fills are light with dark text either way, so subgraph titles and edge labels, which
  have no fill behind them, come out dark grey on near black in a dark theme.
* The library is attached with `add_script_tag(path=...)` rather than inlined. Inlining
  several megabytes of JavaScript into a scratch page works on some machines and times
  out on others.

The source is read from the fence labelled `mermaid-source` in the README, so the
diagram and the document cannot drift apart.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
OUT = ROOT / "docs" / "evidence"

MERMAID_CANDIDATES = (
    ROOT / "node_modules" / "mermaid" / "dist" / "mermaid.min.js",
    Path("/opt/npm-tools/node_modules/mermaid/dist/mermaid.min.js"),
)

FENCE = re.compile(r"^```mermaid-source\n(.*?)^```", re.MULTILINE | re.DOTALL)
BACKGROUND = "#fbfcfd"


def find_mermaid() -> Path:
    for candidate in MERMAID_CANDIDATES:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "mermaid is not available. Install it with `npm install mermaid` in the "
        "repository root, or point MERMAID_CANDIDATES at an existing copy. The "
        "committed SVG is already in docs/evidence, so this is only needed to "
        "regenerate it."
    )


def read_source() -> str:
    match = FENCE.search(README.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit("no ```mermaid-source fence found in README.md")
    return match.group(1)


def render(source: str, library: Path) -> str:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as play:
        browser = play.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.set_content("<!doctype html><html><body><div id='target'></div></body></html>")
        page.add_script_tag(path=str(library))
        svg = page.evaluate(
            """async (source) => {
                mermaid.initialize({
                    startOnLoad: false,
                    htmlLabels: false,
                    flowchart: { htmlLabels: false, curve: 'basis' },
                    theme: 'neutral',
                    securityLevel: 'strict',
                });
                const { svg } = await mermaid.render('diagram', source);
                return svg;
            }""",
            source,
        )
        browser.close()
    return svg


def fix_root(svg: str) -> str:
    """Give the root intrinsic dimensions from its viewBox and paint a background."""
    box = re.search(r'viewBox="([\d.\-\s]+)"', svg)
    if box is None:
        raise SystemExit("the rendered SVG has no viewBox, so it cannot be sized")
    _, _, width, height = (float(v) for v in box.group(1).split())
    svg = re.sub(r'\swidth="[^"]*"', "", svg, count=1)
    svg = re.sub(r'\sheight="[^"]*"', "", svg, count=1)
    svg = re.sub(r'\sstyle="[^"]*max-width[^"]*"', "", svg, count=1)
    svg = svg.replace(
        "<svg ", f'<svg width="{width:.0f}" height="{height:.0f}" ', 1
    )
    rect = (
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="{BACKGROUND}"/>'
    )
    index = svg.index(">", svg.index("<svg")) + 1
    return svg[:index] + rect + svg[index:]


def verify(path: Path) -> dict:
    """Load the committed file over file:// and assert the browser gives it a size.

    Not `set_content`: a page served from about:blank has an opaque origin and will
    not load a local file as a subresource, so the check would pass on a file that is
    broken in the README.
    """
    from playwright.sync_api import sync_playwright

    probe = OUT / "_diagram_probe.html"
    probe.write_text(
        f"<!doctype html><html><body><img id='d' src='{path.name}'></body></html>",
        encoding="utf-8",
    )
    try:
        with sync_playwright() as play:
            browser = play.chromium.launch()
            page = browser.new_page()
            page.goto(probe.resolve().as_uri())
            page.wait_for_timeout(400)
            size = page.evaluate(
                "() => { const i = document.getElementById('d');"
                " return {w: i.naturalWidth, h: i.naturalHeight}; }"
            )
            browser.close()
    finally:
        probe.unlink(missing_ok=True)
    if not size["w"] or not size["h"]:
        raise SystemExit(
            f"{path.name} loads with zero natural width or height, so it will show as a "
            "broken image in the README. Check htmlLabels and the root dimensions."
        )
    return size


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    library = find_mermaid()
    svg = fix_root(render(read_source(), library))
    target = OUT / "architecture.svg"
    target.write_text(svg, encoding="utf-8")
    size = verify(target)
    manifest = {
        "file": target.name,
        "source": "the ```mermaid-source fence in README.md",
        "renderer": str(library),
        "html_labels": False,
        "background": BACKGROUND,
        "natural_size": size,
    }
    (OUT / "diagram-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {target} at {size['w']}x{size['h']} and verified it loads")
    return 0


if __name__ == "__main__":
    sys.exit(main())
