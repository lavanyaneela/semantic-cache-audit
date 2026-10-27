# ADR-005: What this repository cannot measure

**Status:** accepted
**Context:** the boundary, stated as its own decision rather than as a disclaimer

## Decision

Name the claims that are out of reach here, name what each would need, and build the
tool so it refuses rather than guesses when it reaches one.

## One: a budget tighter than the corpus can resolve

This is the boundary that is wired into the code. With 120 positive controls, the
tightest budget provable at 95% confidence is 2.47%, because a flawless run of that
many clean hits bounds the rate no tighter. A 1% budget needs 299 clean hits, which
would take about 798 pairs.

So the tool returns exit code 2 and prints what the budget would require. It does this
for every embedder including a perfectly separating one, because the limit is a
property of the evidence and not of the model. A measured zero over a small sample is
not zero, it is below one in N, and approving on it would be the single most
dishonest thing this repository could do.

## Two: what any particular encoder does on production traffic

Every rate here is a fact about one encoder over one corpus. The corpus is adversarial
by construction, so the unweighted rates describe an adversarial stream. The
prevalence weighted rates are reported beside them, and the prevalence is **declared,
not observed**: nobody here has measured how often a real cache sees a near miss.

What this would need: a labelled sample of real adjacent query pairs from a production
log, with answers judged by someone who knows the domain. That is the one measurement
that would turn the published rates into a forecast.

## Three: how an approximate index changes the answer

The index here is exact, on purpose, so that a false hit is attributable to the
threshold alone. A production approximate index moves the number in both directions
and introduces a failure mode this repository does not have. These figures are a lower
bound on an approximate index rather than an estimate of one. See ADR-003.

## Four: real Redis and MongoDB behaviour

A non blocking CI job runs these code paths against real servers. Because it is non
blocking its failure is invisible in the badge, so the honest form is "a job exists
that exercises these paths" rather than "the paths pass". Nothing here says anything
about either system under load, about eviction interacting with a real memory limit,
or about replication lag.

One defect found by audit and worth recording: the hot layer and the payload store
were global while the vector store was namespaced, so against a shared server two
caches would have become one. That was invisible against the in process doubles, and
it is the kind of thing only a real server finds.

## Five: whether the separation family resembles any real model

It does not claim to. It is an instrument that spans a range of answer separation so
the decision table can be validated across it, and every figure taken from it is
printed with its noise level. A reader who treats a `synthetic@` number as a
measurement of a model has been misled, which is why a bare `synthetic` name is
refused by the CLI.

## Six: latency

The cache's entire value is a latency saving, and this repository publishes no
duration at all. A millisecond is a fact about the runner that measured it, so the
claim lives in a test that asserts an inequality rather than in prose that quotes a
figure. `tools/check_durations.py` sweeps every tracked file rather than the checked
documents, because the promise is about the repository.

The consequence is honest and worth stating: this repository can tell a team how wrong
their cache is, and cannot tell them whether the saving is worth it. That second half
needs their traffic and their hardware.

## What would move each of these

| Out of reach | What it needs |
| --- | --- |
| A tighter budget | A larger corpus. The tool computes how much larger. |
| Production rates | Labelled near miss pairs from a real log |
| Approximate index effects | The same audit over an IVF or HNSW index |
| Backend behaviour under load | A soak test against real servers, outside `make verify` |
| Whether the saving is worth it | The team's own traffic and hardware |
