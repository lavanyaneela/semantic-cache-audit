"""Records a replay of real command output, not a screen recording.

Each command is run for real, its stdout, exit code and wall time captured, and a
self contained player types the command and reveals that captured output paced by
that command's measured duration. Playwright records the page, ffmpeg transcodes to
an MP4 and a palette GIF that autoplays inline on GitHub.

Every line of terminal text in the result is real captured output. None is written by
hand, and the README says so.

Two details that matter. The imports are warmed with a setup command before capture,
or the first segment pays for the interpreter start and the pacing is wrong. And the
segments are ordered as the argument rather than as the feature list: the cache
answering confidently first, then the audit saying it is wrong, then the refusal, then
the receipts.

Durations are used only for pacing and are recorded in the manifest. None is
published, per the rule `tools/check_durations.py` enforces.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "video"
CLI = [sys.executable, "-m", "semantic_cache_audit.cli"]
EMBEDDER = "st:sentence-transformers/all-MiniLM-L6-v2"

SEGMENTS: list[tuple[str, list[str]]] = [
    (
        "semantic-cache-audit corpus",
        [*CLI, "corpus"],
    ),
    (
        "semantic-cache-audit classes --budget 0.05",
        [*CLI, "classes", "--embedder", EMBEDDER, "--budget", "0.05"],
    ),
    (
        "semantic-cache-audit audit --budget 0.05",
        [*CLI, "audit", "--embedder", EMBEDDER, "--budget", "0.05"],
    ),
    (
        "semantic-cache-audit audit --budget 0.001",
        [*CLI, "audit", "--embedder", EMBEDDER, "--budget", "0.001"],
    ),
    (
        "semantic-cache-audit compare",
        [*CLI, "compare", "--embedders", EMBEDDER, "bow", "lsa", "charngram"],
    ),
    (
        "python tools/check_numbers.py --prove",
        [sys.executable, "tools/check_numbers.py", "--prove"],
    ),
]

MAX_LINES = 26


@dataclass
class Capture:
    label: str
    argv: list[str]
    exit_code: int
    seconds: float
    output: str


def warm() -> None:
    subprocess.run([*CLI, "--help"], cwd=ROOT, capture_output=True, check=False)


def run_all() -> list[Capture]:
    captures: list[Capture] = []
    for label, argv in SEGMENTS:
        started = time.perf_counter()
        done = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - started
        text = (done.stdout or "") + (done.stderr or "")
        lines = text.rstrip("\n").splitlines()
        if len(lines) > MAX_LINES:
            kept = lines[:MAX_LINES]
            kept.append(f"... {len(lines) - MAX_LINES} more lines")
            lines = kept
        captures.append(
            Capture(label, argv, done.returncode, round(elapsed, 3), "\n".join(lines))
        )
    return captures


PLAYER = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;background:#0f1419;}
#t{width:1200px;height:760px;padding:26px 30px;box-sizing:border-box;
  font:15px/1.5 "DejaVu Sans Mono",monospace;color:#d8e0e8;overflow:hidden;}
.cmd{color:#7fd4a0;white-space:pre-wrap;}
.cmd b{color:#6aa6ff;}
.out{color:#c3ccd5;white-space:pre-wrap;}
.code0{color:#7fd4a0;} .code1{color:#ffb347;} .code2{color:#ff8c69;}
.sp{height:12px;}
</style></head><body><div id="t"></div><script>
const SEGMENTS = __DATA__;
const term = document.getElementById('t');
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
function add(cls, text){const d=document.createElement('div');d.className=cls;d.textContent=text;
  term.appendChild(d);term.scrollTop=term.scrollHeight;return d;}
async function typeCmd(text){const d=add('cmd','$ ');
  for(const ch of text){d.textContent += ch; await sleep(14);} await sleep(160);}
async function reveal(text, pace){const lines=text.split('\\n');
  const per=Math.max(18, Math.min(110, (pace*1000)/Math.max(lines.length,1)));
  for(const line of lines){add('out', line); await sleep(per);}}
(async () => {
  for(const s of SEGMENTS){
    term.innerHTML='';
    await typeCmd(s.label);
    await reveal(s.output, s.seconds);
    add('code'+(s.exit_code>2?1:s.exit_code), 'exit ' + s.exit_code);
    await sleep(1500);
  }
  window.__done = true;
})();
</script></body></html>"""


def build_player(captures: list[Capture]) -> Path:
    data = json.dumps([asdict(c) for c in captures])
    path = OUT / "player.html"
    path.write_text(PLAYER.replace("__DATA__", data), encoding="utf-8")
    return path


def record(player: Path, captures: list[Capture]) -> Path:
    from playwright.sync_api import sync_playwright

    budget = sum(c.seconds for c in captures) + 2.2 * len(captures) + 6.0
    with sync_playwright() as play:
        browser = play.chromium.launch()
        context = browser.new_context(
            viewport={"width": 1200, "height": 760},
            record_video_dir=str(OUT / "_raw"),
            record_video_size={"width": 1200, "height": 760},
        )
        page = context.new_page()
        page.goto(player.resolve().as_uri())
        deadline = time.perf_counter() + budget
        while time.perf_counter() < deadline:
            if page.evaluate("() => window.__done === true"):
                page.wait_for_timeout(900)
                break
            page.wait_for_timeout(300)
        video = page.video
        raw = Path(video.path()) if video else None
        context.close()
        browser.close()
    if raw is None or not raw.exists():
        raise SystemExit("playwright produced no video")
    return raw


def transcode(raw: Path) -> dict:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed, so the MP4 and GIF cannot be produced")
    mp4 = OUT / "demo.mp4"
    gif = OUT / "demo.gif"
    palette = OUT / "_palette.png"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
         "-vf", "scale=1100:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(mp4)],
        check=True,
    )
    common = "fps=9,scale=900:-1:flags=lanczos"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
         "-vf", f"{common},palettegen=max_colors=96", str(palette)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-i", str(palette),
         "-lavfi", f"{common}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3", str(gif)],
        check=True,
    )
    palette.unlink(missing_ok=True)
    return {"mp4_bytes": mp4.stat().st_size, "gif_bytes": gif.stat().st_size}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    warm()
    captures = run_all()
    player = build_player(captures)
    raw = record(player, captures)
    sizes = transcode(raw)
    shutil.rmtree(OUT / "_raw", ignore_errors=True)
    player.unlink(missing_ok=True)
    manifest = {
        "note": (
            "every line of terminal text in the recording is real captured output. "
            "The video is a replay of that output paced by each command's measured "
            "wall time, not a screen recording. Durations appear here for pacing and "
            "are published nowhere."
        ),
        "commands": [
            {"label": c.label, "argv": c.argv, "exit_code": c.exit_code, "seconds": c.seconds}
            for c in captures
        ],
        **sizes,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"recorded {len(captures)} segments")
    print(f"  demo.gif {sizes['gif_bytes'] / 1e6:.2f} MB")
    print(f"  demo.mp4 {sizes['mp4_bytes'] / 1e6:.2f} MB")
    for c in captures:
        print(f"  exit {c.exit_code}  {c.label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
