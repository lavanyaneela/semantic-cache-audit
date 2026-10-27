"""The cache itself: a hot key value layer, a payload store, and the matching policy.

Both backends run against in-process doubles by default so `make verify` needs no
network, and against real servers in an optional CI job so the claim that the code
paths work is not theoretical. The split is the same shape used in project one and is
recorded in ADR-003.
"""
