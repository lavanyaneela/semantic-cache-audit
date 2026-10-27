# ADR-001: A hand built adversarial corpus, not a sampled one

**Status:** accepted
**Context:** the fixture every published figure in this repository is computed from

## Decision

Author the corpus by hand: 60 template families across 6 domains, expanded by
enumerated substitution into 320 labelled pairs, of which 200 are near misses in five
classes and 120 are true paraphrase positive controls.

## The alternative, and why it was rejected

The obvious alternative is to sample adjacent queries from a real query log and label
them. It was rejected because the region this repository measures is empty in sampled
data.

A semantic cache fails when two queries sit close together in embedding space and have
different correct answers. Users do not produce that pattern on purpose. Sample a log
and you get overwhelmingly either unrelated queries, which no threshold would ever
match, or genuine repeats, which any threshold matches correctly. The dangerous middle
is rare enough that a sample of realistic size contains almost none of it, so the
false hit rate measured on sampled data is a measurement of how rare the problem is in
that sample rather than of how badly the cache handles it.

Authoring the pairs puts the measurement where the failure lives. The five near miss
classes were chosen because each is a different mechanism by which two queries can be
lexically and semantically adjacent while needing different answers: direction
reversal, negation, entity swap, numeric change, and scope change.

## What this costs, stated plainly

**The corpus is small, and the resolution floor is therefore large.** This is the cost
and it is not a small one. With 120 positive controls, a flawless run bounds the false
hit rate no tighter than 2.47% at 95% confidence. A team that wants a 1% budget needs
299 clean hits, so roughly 798 pairs would be required, two and a half times what is
here. The tool does not paper over this: it returns exit code 2 and prints the corpus
size the budget would need. ADR-005 carries the rest of the boundary.

**The pairs are not production traffic.** They are adversarial by construction, so a
rate read straight off the corpus describes an adversarial stream. Every rate is
therefore also reported reweighted to a declared near miss prevalence, and the
prevalence is an input rather than an observation.

**Authored substitution is still authored.** Each family lists its substitutions by
hand, so an expanded pair is an authored pair rather than a sample, but the families
constrain the variety. A reviewer should read `fixtures/corpus/pairs.jsonl` and judge
whether the awkward cases are there, which is why every pair carries its template, its
substitution, and a note saying why it is adversarial.

## The positive controls are the part that is easy to get wrong

Without them the false hit curve has no denominator. With only easy ones it has a
flattering denominator: a corpus of near misses plus obvious repeats makes any
embedder look separable. So the positive controls are deliberately mixed in
difficulty, and the majority are hard ones with low lexical overlap, pairs like asking
how to claim back a train fare against asking for the reimbursement procedure for one.
Those are exactly the paraphrases a semantic cache is bought to catch.

A consequence worth stating because it reads as a flaw and is not: a hard paraphrase
set makes this corpus harder on an embedder than a production stream would be. The
corpus measures the gap between the hardest same answer pairs and the hardest
different answer pairs, which is the gap a threshold has to live in.

## Consistency

"One answer serves both" is an equivalence relation, so the corpus must not contain a
chain of paraphrase pairs joining the two sides of a near miss. The loader computes
answer classes by union find and refuses a corpus that contradicts itself. The first
implementation assigned answers pair by pair and could not express the property at
all; `tests/test_regression.py` constructs a contradiction and shows the difference.
