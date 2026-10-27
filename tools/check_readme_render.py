"""Renders the README the way GitHub does and asserts every local image loads.

python-markdown disagrees with GitHub about exactly this kind of file in several
ways, so the rendering uses `cmarkgfm`, which is the library GitHub itself uses. The
page is then loaded in a browser and every image is checked for a non zero natural
size, because a broken image in a README is the first thing a reviewer notices and
nothing else in the toolchain can see it.

Remote badges are reported but not failed on: a network blip is not a defect in this
repository, and `make verify` has to pass offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        import cmarkgfm
        from cmarkgfm.cmark import Options
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        print(f"skipping: {exc}. Install the dev extra to run this check.")
        return 0

    html = cmarkgfm.markdown_to_html_with_extensions(
        (ROOT / "README.md").read_text(encoding="utf-8"),
        options=Options.CMARK_OPT_UNSAFE,
        extensions=["table", "autolink", "strikethrough", "tagfilter"],
    )
    probe = ROOT / "_readme_render_probe.html"
    probe.write_text("<!doctype html><meta charset='utf-8'>" + html, encoding="utf-8")
    try:
        with sync_playwright() as play:
            browser = play.chromium.launch()
            page = browser.new_page()
            page.goto(probe.resolve().as_uri())
            page.wait_for_timeout(900)
            images = page.evaluate(
                "() => [...document.images].map(i => ({"
                " src: i.getAttribute('src'), w: i.naturalWidth, h: i.naturalHeight }))"
            )
            browser.close()
    finally:
        probe.unlink(missing_ok=True)

    broken = []
    for image in images:
        remote = image["src"].startswith("http")
        ok = remote or (image["w"] and image["h"])
        print(f"{'ok ' if ok else 'BAD'} {image['w']}x{image['h']}  {image['src']}")
        if not ok:
            broken.append(image["src"])
    if broken:
        print(f"\n{len(broken)} image(s) will render broken in the README: {broken}",
              file=sys.stderr)
        return 1
    print(f"\nall {len(images)} README images resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
