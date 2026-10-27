# ADR-002: A real pretrained encoder, plus a panel, plus an instrument

**Status:** accepted
**Context:** every threshold this repository publishes is a fact about one embedder

## Decision

Measure a real, widely deployed sentence encoder, `all-MiniLM-L6-v2`, as the headline.
Run the same audit over three lexical embedders implemented in this repository, so a
threshold can be watched failing to transfer between real families. Add a seeded
separation family as an instrument, clearly labelled as not an embedder.

Embed once, commit the matrices, and have everything downstream read a matrix.

## Why the vectors are committed rather than computed

A reviewer with no weights, no account and no network reproduces every published
number with one command. That is worth more here than realism, because the claim this
repository makes is "these numbers are checkable" and a number that needs a download
is not.

The practical consequence is a clean split. `make verify` reads
`fixtures/embeddings/`, and CI sets `HF_HUB_OFFLINE` so that a regression which
reintroduces a download fails loudly instead of passing quietly.
`make embeddings-live` is the only command in the repository that reaches outside it.

## Why ONNX Runtime rather than the sentence-transformers package

The package pulls a deep learning framework this repository otherwise has no use for,
and the pooling it applies to this model is three lines: mean over the attention mask,
then normalise. Running the graph directly keeps the dependency list honest about what
is actually used. Getting the pooling wrong would be silent, so a test recomputes the
committed matrix from the weights when they happen to be cached and compares, and
another checks that padding does not leak into a pooled vector.

## The panel, and what each member is for

`bow`, `lsa` and `charngram` are unigram TF-IDF, unigram plus bigram TF-IDF reduced by
SVD, and hashed character n-grams under a seeded signed projection. They are genuine
functions of the text and they are measured, not decorative: across the 12
transfers between the four real embedders the threshold shifts by 32.2 cosine points
at the median, and that figure only exists because there is more than one real
embedder here. The separation family adds 144 further transfers to the same sweep,
and those are kept out of the published figure on purpose: pooling them would make
the result look thirteen times larger and would make it a statement about an
instrument.

Two traps they pin. Their vocabulary and weights are fitted on this corpus, so their
numbers are an upper bound on what a cold vocabulary would manage rather than a lower
bound. And the character n-gram embedder hashes with sha1 rather than Python's built
in `hash`, which is salted per process: a fixture built on a salted hash is
reproducible exactly once and the failure is silent.

## The separation family is an instrument, not an embedder

`synthetic@<noise>` mixes a shared background direction, an answer vector shared by
every query with the same answer, and a per query noise vector whose weight is the
name. At noise zero it is an oracle; as the noise rises its separation falls to
chance. Nothing about it is a claim about any model.

It exists for one job: the audit's conclusions are a function of the corpus, the
budget, and one property of the embedder, which is how well it separates same answer
pairs from different answer pairs. The family spans that property, so the decision
table in E4 can be validated across the whole range rather than at the single point a
real encoder happens to occupy. Every figure taken from it is printed with its name,
which carries its noise level, and `synthetic` on its own is refused precisely so a
number cannot lose the parameter it depends on.

## What a different encoder would change, and what it would not

It would change the threshold and every rate at that threshold. It would not change
the refusal: with 120 positive controls no embedder can certify a budget tighter than
2.47%, which is a statement about the corpus. Running the audit against another
encoder is `--embedder st:<model>` and a regenerated fixture.
