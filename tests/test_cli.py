"""Properties of the command line surface, especially the exit codes."""

from __future__ import annotations

import json

import pytest

from semantic_cache_audit.audit.budget import Exit
from semantic_cache_audit.cli import main
from semantic_cache_audit.embed.registry import PRETRAINED_NAME

HEADLINE = ["--embedder", PRETRAINED_NAME]


def test_the_audit_command_runs_and_prints_a_verdict(capsys):
    main(["audit", *HEADLINE, "--budget", "0.05"])
    assert "verdict" in capsys.readouterr().out


def test_a_budget_the_corpus_cannot_resolve_exits_two(capsys):
    assert main(["audit", *HEADLINE, "--budget", "0.001"]) == int(Exit.REFUSED)


def test_a_resolvable_budget_no_threshold_meets_exits_one(capsys):
    assert main(["audit", *HEADLINE, "--budget", "0.05"]) == int(Exit.NO_THRESHOLD)


def test_a_budget_a_threshold_meets_exits_zero(capsys):
    assert main(["audit", "--embedder", "synthetic@0", "--budget", "0.05"]) == int(Exit.OK)


def test_a_refusal_prints_the_corpus_size_it_would_need(capsys):
    main(["audit", *HEADLINE, "--budget", "0.001"])
    assert "positive control" in capsys.readouterr().out


def test_the_audit_command_can_emit_json(capsys):
    main(["audit", *HEADLINE, "--budget", "0.05", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"]["exit_name"] == "NO_THRESHOLD"


def test_the_json_payload_carries_the_whole_curve(capsys):
    main(["audit", *HEADLINE, "--budget", "0.05", "--steps", "11", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["sweep"]["points"]) == 11


def test_an_unknown_embedder_exits_three(capsys):
    assert main(["audit", "--embedder", "nonsense"]) == int(Exit.ERROR)


def test_an_unknown_embedder_explains_itself_on_stderr(capsys):
    main(["audit", "--embedder", "nonsense"])
    assert "unknown embedder" in capsys.readouterr().err


def test_a_malformed_synthetic_name_exits_three(capsys):
    assert main(["audit", "--embedder", "synthetic@banana"]) == int(Exit.ERROR)


def test_an_embedder_with_no_fixture_exits_three(capsys):
    assert main(["audit", "--embedder", "synthetic@7.77"]) == int(Exit.ERROR)


def test_the_sweep_command_prints_the_curve(capsys):
    main(["sweep", *HEADLINE, "--every", "25"])
    assert "wrong share" in capsys.readouterr().out


def test_the_classes_command_prints_a_breakdown(capsys):
    main(["classes", *HEADLINE, "--budget", "0.05"])
    assert "direction_reversal" in capsys.readouterr().out


def test_the_classes_command_says_so_when_there_is_no_usable_threshold(capsys):
    main(["classes", *HEADLINE, "--budget", "0.001", "--at", "operating-point"])
    assert "no usable threshold" in capsys.readouterr().out


def test_the_classes_command_defaults_to_the_reference_threshold(capsys):
    """At the budget operating point on an inverted embedder every class sits at one
    hundred percent, so the default has to be the threshold E2 measures at."""
    main(["classes", *HEADLINE])
    out = capsys.readouterr().out
    assert "reference threshold" in out and "half the true paraphrases" in out


def test_the_classes_command_reproduces_the_per_class_figures_the_readme_quotes(capsys):
    main(["classes", *HEADLINE])
    out = capsys.readouterr().out
    assert "92.50%" in out and "85.00%" in out and "15.00%" in out


def test_the_default_embedder_is_the_one_the_documents_quote():
    """These disagreed once, and every command in the defense guide printed figures
    the documents did not contain."""
    from semantic_cache_audit.config import DEFAULTS

    assert DEFAULTS.embedder == PRETRAINED_NAME


def test_the_bare_audit_command_reproduces_the_headline_separation(capsys):
    main(["audit", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert round(payload["sweep"]["separation_auc"], 3) == 0.238


def test_the_compare_command_lists_every_embedder(capsys):
    main(["compare", "--budget", "0.05"])
    out = capsys.readouterr().out
    assert PRETRAINED_NAME in out and "charngram" in out


def test_the_compare_command_exits_zero_because_it_is_informational(capsys):
    assert main(["compare", "--budget", "0.05"]) == int(Exit.OK)


def test_the_compare_command_accepts_an_explicit_list(capsys):
    main(["compare", "--embedders", "bow", "lsa"])
    out = capsys.readouterr().out
    assert "charngram" not in out


def test_the_corpus_command_prints_the_manifest(capsys):
    main(["corpus"])
    assert "corpus_pairs_total" in capsys.readouterr().out


def test_the_corpus_command_can_emit_json(capsys):
    main(["corpus", "--json"])
    assert json.loads(capsys.readouterr().out)["corpus_pairs_total"] == 320


def test_the_corpus_command_exits_zero(capsys):
    assert main(["corpus"]) == int(Exit.OK)


def test_the_report_command_writes_a_file(tmp_path, capsys):
    out = tmp_path / "report.html"
    main(["report", *HEADLINE, "--out", str(out)])
    assert out.exists() and "<html" in out.read_text()


def test_the_report_carries_the_verdict(tmp_path, capsys):
    out = tmp_path / "report.html"
    main(["report", *HEADLINE, "--budget", "0.001", "--out", str(out)])
    assert "REFUSED" in out.read_text()


def test_the_report_names_every_heading_the_screenshot_tool_frames_by(tmp_path, capsys):
    out = tmp_path / "report.html"
    main(["report", *HEADLINE, "--budget", "0.05", "--out", str(out)])
    text = out.read_text()
    for heading in ("id='verdict'", "id='operating-point'", "id='by-class'", "id='curve'",
                    "id='notes'"):
        assert heading in text


def test_an_unexpected_failure_exits_three_rather_than_crashing(capsys):
    assert main(["audit", *HEADLINE, "--steps", "0"]) == int(Exit.ERROR)


def test_an_unexpected_failure_suggests_the_debug_switch(capsys):
    main(["audit", *HEADLINE, "--steps", "0"])
    assert "SCA_DEBUG" in capsys.readouterr().err


def test_the_debug_switch_re_raises_so_a_real_bug_is_debuggable(monkeypatch):
    """Catching broadly makes a genuine bug look like a user error, so there has to
    be a way back to the traceback."""
    monkeypatch.setenv("SCA_DEBUG", "1")
    with pytest.raises(ValueError, match="--steps"):
        main(["audit", *HEADLINE, "--steps", "0"])


def test_an_out_of_range_prevalence_is_refused(capsys):
    assert main(["audit", *HEADLINE, "--prevalence", "5"]) == int(Exit.ERROR)


def test_an_out_of_range_minimum_hit_rate_is_refused(capsys):
    assert main(["audit", *HEADLINE, "--min-hit-rate", "7"]) == int(Exit.ERROR)


def test_the_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        main([])


def test_the_help_text_documents_the_exit_codes(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "Exit 0" in capsys.readouterr().out


def test_the_prevalence_is_adjustable_from_the_command_line(capsys):
    main(["audit", *HEADLINE, "--budget", "0.05", "--prevalence", "0.5", "--json"])
    payload = json.loads(capsys.readouterr().out)
    point = payload["decision"]["operating_point"]
    assert point["weighted_false_hit_rate"] > 0.4
