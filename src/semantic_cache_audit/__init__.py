"""Semantic cache audit.

An approximate-match cache returns an answer that is indistinguishable from a correct
one at the call site, so a hit rate metric cannot contain the cost of a wrong hit. This
package measures the cost directly: it sweeps similarity thresholds over an adversarial
near-miss corpus and reports the share of hits that return the wrong answer.
"""

__version__ = "1.0.0"
