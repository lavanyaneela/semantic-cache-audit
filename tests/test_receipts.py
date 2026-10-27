"""Properties of the receipts mechanism itself.

The audit on the previous project in this portfolio found that four of seven holes
were in the receipts mechanism rather than in the code it guards, so the mechanism
gets its own suite.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from check_durations import DURATION, scan  # noqa: E402
from check_numbers import (  # noqa: E402
    EXEMPT_NUMBERS,
    anchor_pattern,
    check_exemptions_hide_nothing,
    normalise,
    run,
    strip_for_reverse,
)
from metrics_spec import METRICS, by_name  # noqa: E402


@pytest.fixture(scope="module")
def payload():
    path = ROOT / "docs" / "metrics.json"
    if not path.exists():
        pytest.skip("docs/metrics.json has not been generated yet")
    return json.loads(path.read_text())


def test_every_metric_has_at_least_one_anchor():
    assert all(m.anchors for m in METRICS)


def test_no_two_metrics_share_an_anchor_even_by_containment():
    flat = [(m.name, a) for m in METRICS for a in m.anchors]
    clashes = [
        (n1, a1, n2, a2)
        for i, (n1, a1) in enumerate(flat)
        for n2, a2 in flat[i + 1 :]
        if n1 != n2 and (a1 in a2 or a2 in a1)
    ]
    assert clashes == []


def test_every_anchor_names_the_number_and_only_a_few_words():
    """Receipts rule two: a longer phrase straddles a line break and matches nothing."""
    for metric in METRICS:
        for anchor in metric.anchors:
            assert len(anchor.replace("{}", "").split()) <= 4, (metric.name, anchor)


def test_every_metric_name_is_unique():
    names = [m.name for m in METRICS]
    assert len(names) == len(set(names))


def test_no_exemption_hides_a_measured_value(payload):
    assert check_exemptions_hide_nothing(payload) == []


def test_every_exemption_carries_a_reason():
    assert all(len(reason) > 10 for reason in EXEMPT_NUMBERS.values())


def test_the_checked_documents_all_exist(payload):
    for relative in payload["checked_documents"]:
        assert (ROOT / relative).exists(), relative


def test_the_defense_guide_is_checked_as_well_as_the_readme(payload):
    """A guide read aloud in an interview that has drifted is worse than a stale README."""
    assert "docs/defense-guide.md" in payload["checked_documents"]


def test_every_adr_is_a_checked_document(payload):
    on_disk = sorted(p.name for p in (ROOT / "docs" / "adr").glob("ADR-*.md"))
    checked = sorted(Path(d).name for d in payload["checked_documents"] if "/adr/" in d)
    assert on_disk == checked


def test_the_documents_are_currently_consistent_with_the_code(payload):
    assert [str(f) for f in run(payload)] == []


def test_an_anchor_matches_across_a_line_break():
    pattern = anchor_pattern("{} labelled pairs", False)
    assert pattern.search("over 320\nlabelled pairs here")


def test_an_anchor_matches_a_bolded_figure():
    assert anchor_pattern("{} tests", False).search("**426** tests")


def test_an_anchor_matches_a_thousands_separator():
    assert anchor_pattern("{} tests", False).search("1,426 tests")


def test_a_textual_anchor_matches_a_whole_token_not_a_fragment():
    """The slot was non greedy once and matched single letters out of nearby prose."""
    pattern = anchor_pattern("worst class is {}", True)
    match = pattern.search("the worst class is direction_reversal, at")
    assert match and match.group(1) == "direction_reversal"


def test_a_textual_anchor_does_not_match_a_different_phrase():
    pattern = anchor_pattern("{} cuts staleness", True)
    match = pattern.search("Evicting by age cuts staleness")
    assert match is None or match.group(1) != "fifo"


def test_the_reverse_pass_ignores_fenced_blocks():
    assert "99999" not in strip_for_reverse("text\n```\n99999\n```\nmore")


def test_the_reverse_pass_ignores_inline_code():
    assert "99999" not in strip_for_reverse("see `--budget 99999` here")


def test_the_reverse_pass_ignores_html_attributes():
    assert "99999" not in strip_for_reverse('<img width="99999">')


def test_the_reverse_pass_ignores_urls():
    assert "99999" not in strip_for_reverse("https://example.com/99999")


def test_the_reverse_pass_ignores_adr_references():
    assert "004" not in strip_for_reverse("see ADR-004 for the definition")


def test_the_forward_pass_sweeps_inside_fenced_blocks(payload):
    """Guarding prose alone closes mutation and leaves insertion wide open."""
    stale = f"```\n{int(payload['rendered']['tests_total']) + 1} tests\n```"
    pattern = anchor_pattern("{} tests", False)
    assert pattern.search(stale)


def test_rendering_a_metric_is_stable(payload):
    spec = by_name()
    for name, value in payload["metrics"].items():
        assert spec[name].render(value) == payload["rendered"][name]


def test_no_rendered_value_is_empty(payload):
    assert all(v != "" for v in payload["rendered"].values())


def test_normalising_strips_thousands_separators():
    assert normalise(" 1,426 ") == "1426"


def test_the_duration_pattern_catches_a_published_millisecond():
    assert DURATION.search("the median was 140 ms")


def test_the_duration_pattern_ignores_a_comma_before_a_unit():
    """`Math.min(110, ms * 1000)` is not a duration, and it failed the build once."""
    assert not DURATION.search("Math.min(110, ms * 1000)")


def test_the_duration_pattern_still_catches_a_thousands_grouped_duration():
    assert DURATION.search("took 1,200 seconds")


def test_no_duration_is_published_anywhere_in_the_repository():
    assert [str(h) for h in scan()] == []


def test_every_allowed_duration_path_carries_a_reason():
    from check_durations import ALLOWED_PATHS

    assert all(len(reason) > 10 for reason in ALLOWED_PATHS.values())


def test_the_duration_sweep_sees_a_file_that_is_added_but_not_committed():
    """A new file was invisible to the sweep until it was committed, and the receipts
    suite passed locally and then failed on the next fresh clone because of it."""
    import subprocess

    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    from check_durations import tracked_files

    assert len(tracked_files()) == len([line for line in out.stdout.splitlines() if line])


def test_every_allowed_duration_path_still_exists():
    from check_durations import ALLOWED_PATHS

    missing = [p for p in ALLOWED_PATHS if not (ROOT / p).exists()]
    assert missing == []


def test_the_checker_catches_a_deliberately_broken_figure(payload):
    """Proves the mechanism works, in the suite as well as in CI."""
    done = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_numbers.py"), "--prove"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert done.returncode == 0 and "caught:" in done.stdout


def test_the_readme_leads_with_an_unflattering_number():
    """The first screen must not read as marketing."""
    first_screen = (ROOT / "README.md").read_text(encoding="utf-8")[:3000].lower()
    assert "wrong" in first_screen and "cannot" in first_screen
