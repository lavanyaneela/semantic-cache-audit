"""Request and response bodies.

Note what the lookup response does not contain: any indication of whether the answer
is right. That absence is the repository's subject, so it is a deliberate property of
the schema rather than an oversight, and a test asserts it stays that way.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StoreRequest(BaseModel):
    query: str = Field(min_length=1)
    answer_id: str = Field(min_length=1)
    version: int = 1


class LookupRequest(BaseModel):
    query: str = Field(min_length=1)


class BatchLookupRequest(BaseModel):
    """The batch inference path. One encode pass, one index search, N results."""

    queries: list[str] = Field(min_length=1, max_length=512)


class LookupResponse(BaseModel):
    hit: bool
    answer_id: str | None = None
    answer_version: int | None = None
    score: float | None = None
    exact: bool = False


class BatchLookupResponse(BaseModel):
    results: list[LookupResponse]
    hits: int
    misses: int


class StatsResponse(BaseModel):
    threshold: float
    entries: int
    lookups: int
    exact_hits: int
    approx_hits: int
    misses: int
    note: str
