"""A small REST surface, so the cache is exercised the way it would be in production.

Not a product. It exists because a cache measured only through a Python function call
is measured without the thing that makes caches interesting, which is that the call
site is remote, batched, and cannot see inside the answer it was handed.
"""
