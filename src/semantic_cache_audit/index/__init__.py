"""The vector store, behind the interface a hosted service would expose.

The audit never calls FAISS directly. It calls an upsert, query, delete and namespace
surface shaped like a managed vector database, so that the question "would this result
change on a hosted index" has an answer that is about approximation rather than about
rewriting the caller.
"""
