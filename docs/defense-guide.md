# Defense guide: semantic-cache-audit

Every figure here is re-measured by `tools/collect_metrics.py` and checked against
this file by `tools/check_numbers.py`, so a number that has drifted fails the build
rather than being read aloud in an interview.

## The short version, said out loud

> A semantic cache returns an answer that looks identical to a correct one at the call
> site, so no hit rate can tell you what share of those answers were wrong. I built an
> adversarial corpus of 320 labelled pairs, ran a real sentence encoder over it, and
> measured the share directly. The result is that on `all-MiniLM-L6-v2` there is no
> safe threshold at all: exactly 0 of the 120 true paraphrases can be served from
> cache without also serving at least one wrong answer.

Then stop.

The question that usually comes next is "why is the model that bad", and the honest
answer is that it is not bad, it is doing what it was trained to do. Mean pooled
sentence embeddings are built to be robust to word order, and direction reversal is
pure word order. The model is working; the cache is asking it a question it was never
built to answer.

## The claims, each with how it is proved

### Claim 1: the encoder ranks wrong answers above right ones

**Prove it:** `make audit`

**The numbers:** separation of 0.238, where 0.5 is chance. Separation is the
probability that a true paraphrase outscores an adversarial near miss, so below one
half means the ranking is inverted rather than merely weak. The 77 most confident
matches the model makes are all wrong.

**If pushed, "your corpus is rigged to make that happen":** partly, and deliberately,
which is why the positive controls matter more than the near misses. If I had paired
hard near misses with easy paraphrases the result would be meaningless. Most of the
positive controls are hard ones with low lexical overlap, and if anything that makes
the corpus harder on the encoder than production would be. The number to attack is
not the near misses, it is whether the paraphrases are real. Read
`fixtures/corpus/pairs.jsonl` and judge.

### Claim 2: there is no safe threshold, not merely a bad one

**Prove it:** `semantic-cache-audit audit --budget 0.05`, exit code 1.

**The numbers:** 0 of the 120 true paraphrases sit above the highest scoring near
miss. The worst class is direction_reversal at 100.0% of that class wrongly served,
and across that class the similarity is never below 0.989.

**If pushed, "so raise the threshold":** that is the move the measurement rules out.
The highest scoring near miss outscores every genuine paraphrase, so any threshold
that admits one paraphrase has already admitted a wrong answer. Raising the threshold
does not trade correctness against coverage here; it just turns the cache off.

### Claim 3: the tool knows when its own evidence is insufficient

**Prove it:** `semantic-cache-audit audit --budget 0.001`, exit code 2.

**The numbers:** with 120 positive controls, no run can certify a budget tighter than
2.47% at 95% confidence, because that is the Clopper-Pearson bound on a flawless run
of that size. A 1% budget needs 299 clean hits, and about 798 pairs would be needed.
The tool prints that rather than reporting the measured rate.

**If pushed, "isn't exit 2 just a failure":** no, and the distinction is the most
useful thing in the repository. Exit 1 says the cache is not good enough and sends a
team to the embedder. Exit 2 says the evidence cannot settle the question and sends
them to the corpus. A tool that collapses those tells people to do the wrong work.

### Claim 4: a threshold does not transfer between embedders

**Prove it:** `python experiments/e5_threshold_transfer.py`

**The numbers:** across the 12 transfers between the four real embedders, the
threshold shifts by 32.2 cosine points at the median. 5 stop rejecting anything.
Separately, and these are a different three, in 3 transfers every hit the cache serves
is wrong: those are the pretrained encoder's threshold carried onto each of the three
lexical embedders. The same sweep covers the
separation family, which adds 144 further transfers; those are reported separately
and are not part of this claim.

**If pushed, "that is obvious":** it is obvious once stated and it is what teams do
anyway, because the threshold is the one number that is easy to copy out of a blog
post. The contribution is the size of the failure, not the direction.

### Claim 5: cache size is a correctness lever, not just a cost lever

**Prove it:** `python experiments/e3_eviction_sweep.py`

**The numbers:** choosing between LRU, LFU and FIFO at a fixed capacity moves the
wrong answer rate by 0.13 points of wrong answers while moving staleness by 16.8
points of staleness. Bounding the cache at all cuts the false hit rate to 2.9% when
bounded from 5.7% when unbounded, which is 2.0 times the wrong answers for an
unbounded cache.

**If pushed, "why would size change correctness":** a wrong answer needs a wrong
neighbour to still be resident. A smaller cache holds fewer candidates, so fewer
queries find a close-but-wrong one. The brief for this project predicted eviction
would move cost and leave correctness alone, and the measurement split that into two
levers instead.

## Questions that are meant to be hard

**"Isn't this just a wrapper around FAISS?"**

The FAISS call is one line and it is deliberately the boring part: a flat index, exact
search, chosen so that a measured false hit is attributable to the threshold rather
than to an approximation. What is built here is the referee. The corpus with its
labels, the definition of a wrong hit in ADR-004, the decision to qualify a threshold
on an upper confidence bound rather than on the point estimate, the refusal logic and
the ceiling it rests on, and the receipts mechanism that re-measures every published
figure.

To be precise about the split, because it matters: the Clopper-Pearson bound itself is
one call to `scipy.stats.beta.ppf` and the separation statistic is one call to
`scipy.stats.mannwhitneyu`. SciPy does the arithmetic. What is mine is choosing to
bound rather than to estimate, working out that the ceiling on clean hits is the
positive control count, and wiring the consequence into an exit code.

**"The corpus is synthetic. Does any of this transfer?"**

Partly, and the parts divide cleanly. The separation statistic and the class ranking
are properties of the encoder and transfer to any corpus with the same shapes: nothing
about mean pooled embeddings discarding word order is specific to my pairs. The rates
do not transfer, because they depend on how often a real stream contains a near miss,
which nobody here has measured. That is why every rate is also reported reweighted to
a declared prevalence, and why the prevalence is labelled declared.

**"What is the weakest part?"**

The corpus size, and the repository says so in its own exit code. 120 positive
controls cannot certify a budget tighter than 2.47%, which is looser than any
production cache would accept. I could have hidden that by reporting the measured rate
and letting the reader assume it was precise. Instead the tool refuses and prints that
about 798 pairs would be needed. It is still the thing I would fix first.

The second weakest is that the near miss prevalence is declared rather than observed.
Every prevalence weighted number in the repository rests on a figure I chose.

**"Why only one real encoder?"**

Because one is enough to make the claim and more would not change it, while running
four embedders is what lets me show a threshold failing to transfer. The three lexical
embedders are not padding: they are what turns "a threshold is a fact about one
embedding space" from an assertion into 12 transfers between real embedders.
Adding a second
pretrained encoder is one flag and a regenerated fixture, and I would expect its
separation to be better and still below one half.

**If pushed, "the JSON has a bigger transfer count":** the sweep runs over every embedder
fixture including the separation family, and the pooled count is in the JSON. The
published figure is the real only one, because a transfer between two draws from a
seeded family is not evidence about models. I split them after an audit pointed out
that the pooled number was doing work it had not earned.

**"What would you do differently with more time?"**

Grow the corpus to the size the refusal asks for, so the budget question becomes
answerable rather than declined. Then build the cheap class specific guard, a
direction and negation detector in front of the cache, and measure whether it moves
separation more than a larger encoder does. My guess is that it does, and that is a
guess worth being wrong about in public.

**"You wrote the corpus and then measured a model on it. Isn't that circular?"**

It would be if the corpus had been tuned against the model, and it was not: the
templates were written and committed before any pretrained vector existed in the
repository, and the git history shows that order. The sharper version of the question
is whether the five classes are the right five, and the answer is that they are the
five mechanisms I could name by which two queries can be adjacent and need different
answers. If someone names a sixth, the corpus should grow.

**"Why no latency numbers, when latency is the whole point of a cache?"**

Because a millisecond is a fact about the machine that measured it, and this
repository's entire claim is that its numbers are checkable on someone else's machine.
`tools/check_durations.py` sweeps every tracked file to enforce that. The honest
consequence is a limit: I can tell a team how wrong their cache is and not whether the
saving is worth it, and the second half needs their traffic.

## Things to say

- "Separation of 0.238, where 0.5 is chance, so the ranking is inverted."
- "Twelve transfers between real embedders, not the pooled count that includes the
  instrument."
- "No safe threshold, because the best near miss outscores every true paraphrase."
- "The tool has three answers, not two: yes, no, and your evidence cannot tell."
- "Every number in the README is re-measured by CI and fails the build if the prose
  drifts. You can watch it fail with `make prove`."

## Things not to say

- Do not say "semantic caches return wrong answers 60% of the time". That figure is
  the near miss share of an adversarial corpus and means nothing about production.
  Say "on this adversarial corpus, at the threshold that serves half the paraphrases".
- Do not say "zero false hits" about anything. Say zero observed, over N hits, bounded
  at the Clopper-Pearson upper limit.
- Do not say "built on Redis and MongoDB". Say there is a non blocking CI job that
  runs the cache tests against real service containers, that it cannot fail the
  build, and that `make verify` runs against in process doubles so it works offline.
- Do not present any `synthetic@` figure as a measurement of a model. It is an
  instrument that spans a range of answer separation.
- Do not claim the cache is slow or fast. No duration is published anywhere.

## The live demo, five commands

```bash
make audit                                        # exit 1, with the separation and the class table
semantic-cache-audit audit --budget 0.001         # exit 2, the refusal and the corpus size it needs
semantic-cache-audit classes                      # where it goes wrong, by class
semantic-cache-audit compare                      # every embedder side by side
make prove                                        # the number checker failing on purpose
```
