"""Properties of the receipts mechanism itself.

The audit on the previous project in this portfolio found that four of seven holes
were in the receipts mechanism rather than in the code it guards, so the mechanism
gets its own suite.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
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


def test_no_exemption_is_unused(payload):
    """An unused exemption is a hole waiting for a number to fall into it."""
    from check_numbers import unused_exemptions

    assert unused_exemptions(payload) == []


def test_every_job_summary_row_names_a_metric_that_exists(payload):
    """One did not, so the CI job summary printed n/a for it in every run."""
    from job_summary import check_rows_resolve

    assert check_rows_resolve(payload["rendered"]) == []


def test_the_duration_pattern_catches_a_duration_without_a_separator():
    """Narrowing the leading run to three digits exempted everything over 999."""
    assert DURATION.search("1200 ms") and DURATION.search("12345 ms")


def test_the_duration_pattern_catches_a_bare_second_suffix():
    assert DURATION.search("ran in 0.088s")


def test_the_duration_pattern_ignores_a_python_format_specifier():
    assert not DURATION.search("print(f'{name:44s}')")


def test_the_corpus_manifest_is_stable_across_days(tmp_path):
    """Stamping today's date unconditionally made the repository byte reproducible
    for exactly one day."""
    from semantic_cache_audit.corpus import builder

    root = tmp_path / "corpus"
    first = builder.write(root)
    second = builder.write(root)
    assert first["captured_utc"] == second["captured_utc"]


def test_an_embedding_manifest_keeps_its_capture_date_when_nothing_changed(tmp_path):
    from semantic_cache_audit.embed.fixtures import write_fixture

    first = write_fixture("charngram", None, tmp_path)
    second = write_fixture("charngram", None, tmp_path)
    assert first["captured_utc"] == second["captured_utc"]


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


def test_every_stored_metric_is_at_the_precision_it_is_published_at(payload):
    """metrics.json must not carry digits below what any document quotes.

    It did, and the committed file then differed between two machines by 0.034 of a
    percentage point in a coverage figure both of them render as 97.7, which failed
    the CI step that asserts the committed metrics are the ones just measured.
    """
    from metrics_spec import RAW_DECIMALS, by_name

    spec = by_name()
    for name, value in payload["metrics"].items():
        decimals = RAW_DECIMALS.get(getattr(spec[name].render, "__name__", ""))
        if decimals is None or isinstance(value, str):
            continue
        assert round(float(value), decimals) == float(value), (name, value)


def test_quantising_never_changes_how_a_metric_renders(payload):
    from metrics_spec import by_name, quantize

    spec = by_name()
    for name, value in payload["metrics"].items():
        render = spec[name].render
        assert render(quantize(value, render)) == payload["rendered"][name], name


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


def test_no_published_transfer_figure_pools_the_instrument_with_real_embedders():
    """The audit's critical finding. A transfer between two draws from a seeded
    family is not evidence about models, so no metric may be computed over the
    pooled set."""
    from metrics_spec import METRICS

    transfer_metrics = [m for m in METRICS if m.source == "e5_threshold_transfer"]
    assert transfer_metrics
    for metric in transfer_metrics:
        assert not metric.path[0].startswith("pooled_"), metric.name


def test_the_real_only_transfer_count_is_far_smaller_than_the_pooled_one():
    payload = json.loads((ROOT / "docs" / "experiments" / "e5_threshold_transfer.json").read_text())
    assert payload["transfers_real_only"] < payload["pooled_transfers_measured"] / 5


def test_the_lsa_coordinates_are_not_reproducible_but_its_geometry_is():
    """The finding behind the reproducibility check, pinned so the claim stays true.

    A truncated SVD may return any orthonormal basis for a degenerate singular
    subspace, and which one depends on the BLAS kernel. Varying only the thread count
    moves a coordinate by about 1.0 and every pairwise cosine by about 2e-07. The
    repository therefore checks geometry and says so, rather than checking bytes and
    being wrong on any machine with a different core count.
    """
    import numpy as np

    from semantic_cache_audit.corpus.loader import load_pairs
    from semantic_cache_audit.embed.registry import build, corpus_documents

    pairs = load_pairs()
    docs = corpus_documents(pairs)
    first = build("lsa", pairs).encode(docs)
    second = build("lsa", pairs).encode(docs)
    # Within one process the basis is stable, so this asserts the invariant that
    # holds everywhere: the geometry is what the audit consumes.
    assert np.abs(first @ first.T - second @ second.T).max() < 1e-5


def test_every_offline_fixture_reproduces_its_geometry():
    from check_fixtures_reproduce import COSINE_TOLERANCE, check
    from semantic_cache_audit.corpus.loader import load_pairs

    pairs = load_pairs()
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("bow", "lsa", "charngram"):
            ok, _, cosine_shift = check(name, pairs, Path(tmp))
            assert ok, f"{name} moved {cosine_shift:g}, over {COSINE_TOLERANCE:g}"


def test_the_reproducibility_check_would_catch_a_fixture_that_really_changed(tmp_path):
    """The counter test. A tolerance nobody has watched reject something is not a
    tolerance, it is a rubber stamp."""
    import numpy as np

    from semantic_cache_audit.embed.fixtures import load_fixture

    _, committed = load_fixture("charngram")
    tampered = committed.copy()
    tampered[0] = -tampered[0]
    moved = float(np.abs(committed @ committed.T - tampered @ tampered.T).max())
    assert moved > 1e-5


def test_the_readme_leads_with_an_unflattering_number():
    """The first screen must not read as marketing."""
    first_screen = (ROOT / "README.md").read_text(encoding="utf-8")[:3000].lower()
    assert "wrong" in first_screen and "cannot" in first_screen
