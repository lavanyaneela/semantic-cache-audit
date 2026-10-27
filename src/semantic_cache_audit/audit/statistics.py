"""The arithmetic that keeps a measured zero from being read as zero.

Receipts rule four, in code. A cache that produced no wrong answer over thirty hits
has not been shown to be safe; it has been shown to be below roughly one in ten. Every
rate this repository publishes therefore carries an upper confidence bound, and the
decision in `budget.py` is taken against the bound rather than against the point
estimate.
"""

from __future__ import annotations

import math

from scipy import stats

#: One sided confidence used for every published bound. Stated here once so the
#: README, the ADRs and the CLI cannot disagree about it.
CONFIDENCE = 0.95


def upper_bound(successes: int, trials: int, confidence: float = CONFIDENCE) -> float:
    """Clopper-Pearson one sided upper bound on a proportion.

    Exact rather than normal approximation, because the interesting region of this
    repository is small counts over small denominators, which is exactly where the
    normal approximation is worst. At zero successes it reduces to the familiar rule of
    three: the bound is about 3/n.
    """
    if trials <= 0:
        return 1.0
    if successes < 0 or successes > trials:
        raise ValueError(f"{successes} successes out of {trials} trials is not a proportion")
    if successes == trials:
        return 1.0
    return float(stats.beta.ppf(confidence, successes + 1, trials - successes))


def resolution_floor(trials: int) -> float:
    """The smallest non zero rate a sample of this size can express, as a proportion."""
    return 1.0 / trials if trials > 0 else 1.0


def trials_needed_for(budget: float, confidence: float = CONFIDENCE) -> int:
    """How many hits a clean run would need before `budget` could be certified.

    Inverts the zero success bound: with no observed failures the upper bound is
    `1 - (1 - confidence) ** (1 / n)`, so the smallest n that pushes it under the
    budget is the answer the tool prints when it refuses.
    """
    if not 0.0 < budget < 1.0:
        raise ValueError(f"budget must be strictly between 0 and 1, got {budget}")
    return int(math.ceil(math.log(1.0 - confidence) / math.log(1.0 - budget)))


def separation_auc(positive_scores, negative_scores) -> float:
    """Probability that a same-answer pair scores above a different-answer pair.

    One number for the only property of an embedder a cache audit depends on. Half is
    chance. Below half means the embedder ranks a wrong answer above a right one, which
    is not a weak embedder but an inverted one, and no threshold repairs it.
    """
    n_pos, n_neg = len(positive_scores), len(negative_scores)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    result = stats.mannwhitneyu(positive_scores, negative_scores, alternative="greater")
    return float(result.statistic) / (n_pos * n_neg)
