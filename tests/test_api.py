"""Properties of the REST surface, including the one the schema deliberately lacks."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from semantic_cache_audit.api.app import build_app
from semantic_cache_audit.api.models import LookupResponse
from semantic_cache_audit.embed.registry import PRETRAINED_NAME


@pytest.fixture
def client():
    return TestClient(build_app(PRETRAINED_NAME, threshold=0.9))


@pytest.fixture
def reversal(pairs):
    return next(p for p in pairs if p.cls == "direction_reversal")


@pytest.fixture
def paraphrase(pairs):
    return next(p for p in pairs if p.same_answer and p.difficulty == "hard")


def test_health_reports_the_embedder_and_threshold(client):
    body = client.get("/health").json()
    assert body["embedder"] == PRETRAINED_NAME and body["threshold"] == 0.9


def test_storing_an_entry_returns_created(client, reversal):
    response = client.post("/v1/entries", json={"query": reversal.query_a, "answer_id": "A"})
    assert response.status_code == 201


def test_a_stored_query_is_returned_on_an_exact_lookup(client, reversal):
    client.post("/v1/entries", json={"query": reversal.query_a, "answer_id": "A"})
    body = client.post("/v1/lookup", json={"query": reversal.query_a}).json()
    assert body["hit"] and body["exact"] and body["answer_id"] == "A"


def test_an_unstored_query_misses(client, reversal):
    assert client.post("/v1/lookup", json={"query": reversal.query_b}).json()["hit"] is False


def test_a_reversed_route_is_served_the_other_direction_s_answer(client, reversal):
    """The impossibility, in one HTTP exchange.

    The stored answer is for one direction. The reversed query is a different
    question with a different answer, and the response is a hit with a high score and
    no field a caller could inspect to find out.
    """
    client.post("/v1/entries", json={"query": reversal.query_a, "answer_id": "A"})
    body = client.post("/v1/lookup", json={"query": reversal.query_b}).json()
    assert body["hit"] and body["answer_id"] == "A" and body["score"] > 0.98


def test_the_lookup_response_has_no_field_describing_correctness():
    assert not set(LookupResponse.model_fields) & {"correct", "wrong", "valid", "false_hit"}


def test_a_hard_paraphrase_is_not_served_at_a_high_threshold(client, paraphrase):
    client.post("/v1/entries", json={"query": paraphrase.query_a, "answer_id": "A"})
    assert client.post("/v1/lookup", json={"query": paraphrase.query_b}).json()["hit"] is False


def test_a_query_outside_the_fixture_is_refused_rather_than_guessed(client):
    response = client.post("/v1/lookup", json={"query": "a query nobody committed"})
    assert response.status_code == 422


def test_an_empty_query_is_rejected_by_validation(client):
    assert client.post("/v1/lookup", json={"query": ""}).status_code == 422


def test_an_entry_can_be_deleted(client, reversal):
    client.post("/v1/entries", json={"query": reversal.query_a, "answer_id": "A"})
    client.delete(f"/v1/entries/entry:{reversal.query_a}")
    assert client.post("/v1/lookup", json={"query": reversal.query_a}).json()["hit"] is False


def test_the_batch_endpoint_returns_one_result_per_query(client, reversal):
    client.post("/v1/entries", json={"query": reversal.query_a, "answer_id": "A"})
    body = client.post(
        "/v1/lookup/batch", json={"queries": [reversal.query_a, reversal.query_b]}
    ).json()
    assert len(body["results"]) == 2


def test_the_batch_endpoint_counts_hits_and_misses(client, reversal):
    client.post("/v1/entries", json={"query": reversal.query_a, "answer_id": "A"})
    body = client.post(
        "/v1/lookup/batch", json={"queries": [reversal.query_a, reversal.query_b]}
    ).json()
    assert body["hits"] + body["misses"] == 2


def test_the_batch_endpoint_agrees_with_the_single_endpoint(pairs, reversal):
    """The batch path is a separate implementation, so the claim needs this test."""
    queries = [reversal.query_a, reversal.query_b, pairs[5].query_a, pairs[9].query_b]
    batch_app = TestClient(build_app(PRETRAINED_NAME, threshold=0.9))
    single_app = TestClient(build_app(PRETRAINED_NAME, threshold=0.9))
    for app in (batch_app, single_app):
        app.post("/v1/entries", json={"query": reversal.query_a, "answer_id": "A"})
    batched = batch_app.post("/v1/lookup/batch", json={"queries": queries}).json()["results"]
    singles = [single_app.post("/v1/lookup", json={"query": q}).json() for q in queries]
    assert [r["hit"] for r in batched] == [r["hit"] for r in singles]


def test_the_batch_endpoint_agrees_on_answers(pairs, reversal):
    queries = [reversal.query_a, reversal.query_b]
    batch_app = TestClient(build_app(PRETRAINED_NAME, threshold=0.9))
    single_app = TestClient(build_app(PRETRAINED_NAME, threshold=0.9))
    for app in (batch_app, single_app):
        app.post("/v1/entries", json={"query": reversal.query_a, "answer_id": "A"})
    batched = batch_app.post("/v1/lookup/batch", json={"queries": queries}).json()["results"]
    singles = [single_app.post("/v1/lookup", json={"query": q}).json() for q in queries]
    assert [r["answer_id"] for r in batched] == [r["answer_id"] for r in singles]


def test_the_batch_endpoint_accounts_for_an_exact_repeat_as_an_exact_hit(client, reversal):
    """It skipped the hot layer once, so /v1/stats reported no exact hits for any
    batch traffic and the two agreement tests could not see it."""
    client.post("/v1/entries", json={"query": reversal.query_a, "answer_id": "A"})
    body = client.post("/v1/lookup/batch", json={"queries": [reversal.query_a]}).json()
    assert body["results"][0]["exact"] is True
    assert client.get("/v1/stats").json()["exact_hits"] == 1


def test_the_batch_and_single_endpoints_leave_identical_statistics(pairs, reversal):
    queries = [reversal.query_a, reversal.query_b, pairs[5].query_a]
    batch_app = TestClient(build_app(PRETRAINED_NAME, threshold=0.9))
    single_app = TestClient(build_app(PRETRAINED_NAME, threshold=0.9))
    for app in (batch_app, single_app):
        app.post("/v1/entries", json={"query": reversal.query_a, "answer_id": "A"})
    batch_app.post("/v1/lookup/batch", json={"queries": queries})
    for query in queries:
        single_app.post("/v1/lookup", json={"query": query})
    keys = ("lookups", "exact_hits", "approx_hits", "misses")
    batched = {k: batch_app.get("/v1/stats").json()[k] for k in keys}
    singles = {k: single_app.get("/v1/stats").json()[k] for k in keys}
    assert batched == singles


def test_the_batch_endpoint_agrees_on_the_exact_flag(pairs, reversal):
    queries = [reversal.query_a, reversal.query_b]
    batch_app = TestClient(build_app(PRETRAINED_NAME, threshold=0.9))
    single_app = TestClient(build_app(PRETRAINED_NAME, threshold=0.9))
    for app in (batch_app, single_app):
        app.post("/v1/entries", json={"query": reversal.query_a, "answer_id": "A"})
    batched = batch_app.post("/v1/lookup/batch", json={"queries": queries}).json()["results"]
    singles = [single_app.post("/v1/lookup", json={"query": q}).json() for q in queries]
    assert [r["exact"] for r in batched] == [r["exact"] for r in singles]


def test_the_batch_endpoint_rejects_an_empty_list(client):
    assert client.post("/v1/lookup/batch", json={"queries": []}).status_code == 422


def test_the_batch_endpoint_refuses_a_query_outside_the_fixture(client):
    response = client.post("/v1/lookup/batch", json={"queries": ["not in the fixture"]})
    assert response.status_code == 422


def test_stats_report_the_counts_the_cache_keeps(client, reversal):
    client.post("/v1/entries", json={"query": reversal.query_a, "answer_id": "A"})
    client.post("/v1/lookup", json={"query": reversal.query_a})
    body = client.get("/v1/stats").json()
    assert body["entries"] == 1 and body["lookups"] == 1


def test_stats_say_plainly_that_hit_counts_prove_nothing(client):
    assert "nothing about whether" in client.get("/v1/stats").json()["note"]


def test_each_application_gets_its_own_cache(reversal):
    first = TestClient(build_app(PRETRAINED_NAME, threshold=0.9))
    first.post("/v1/entries", json={"query": reversal.query_a, "answer_id": "A"})
    second = TestClient(build_app(PRETRAINED_NAME, threshold=0.9))
    assert second.get("/v1/stats").json()["entries"] == 0


def test_the_openapi_schema_is_generated(client):
    assert client.get("/openapi.json").status_code == 200
