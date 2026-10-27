# ADR-004: What counts as a wrong hit

**Status:** accepted
**Context:** every headline in this repository is a count of wrong hits, so the
definition carries the whole result

## Decision

A hit is wrong when the cached answer would be judged incorrect for the query that was
asked. It is **not** wrong merely because the two queries differ.

Operationally, the corpus carries the judgement. Each pair is labelled with whether one
answer genuinely serves both sides, and that label, not the similarity, is the ground
truth. A hit on a pair labelled as sharing an answer is correct however little the two
queries look alike; a hit on a pair labelled otherwise is wrong however similar they
are.

## Why this needs an ADR rather than a footnote

Three weaker definitions are available and each would change the headline.

**"Different strings, so wrong."** This counts every paraphrase served from cache as a
failure, which makes the false hit rate equal to the hit rate and the cache
definitionally useless. It also inverts the finding: under this definition the
embedder that serves nothing scores best.

**"Different intent, so wrong."** Intent is not observable and would have to be judged,
which puts a judgement call inside the measurement. Worse, it is the same judgement the
cache is being tested on, so the measurement would be grading itself.

**"The user complained, so wrong."** The closest to the truth and unavailable offline.
It is also the definition that makes the impossibility concrete: a wrong hit is
invisible at the call site, so complaints are the production signal, and they arrive
late, sparsely, and attached to the wrong component.

The chosen definition is the only one that is both decidable from the fixture and
aligned with what a user would call a wrong answer.

## The two failure modes are counted separately

A hit can be bad in two ways and the fixes are different, so one number for both would
hide which lever to pull.

A **false hit** returned the answer to a different question. The lever is the
similarity threshold.

A **stale hit** returned the right question's answer from before it changed. The lever
is the time to live.

E3 measures them separately and the separation earns its keep: across the bounding
eviction policies the wrong share moves by 0.16 points of wrong answers while
staleness moves by 16.8 points of staleness. Reported as one figure, that would read
as "eviction matters" and send a team to the wrong knob.

## What is deliberately not counted as a wrong hit

A **miss on a true paraphrase** is a missed opportunity, not an error. It costs the
latency saving the cache was bought for, and it is reported alongside every operating
point, but calling it wrong would let an embedder that caches nothing look safe.

A **vector whose payload has been evicted** is a bug, not a miss, and is counted in its
own bucket rather than absorbed into the miss rate.

An **exact repeat** is excluded from the denominator entirely. See ADR-003.

## The consequence the repository is built around

Note what the lookup result does not contain. There is no field a caller could read to
find out whether the answer is right, because if the cache knew it would not have
returned it. `tests/test_cache.py` asserts that the field set stays that way, since the
moment such a field appears the premise is gone. Everything under `src/.../audit/` is an
external referee with access to labels the cache does not have.
