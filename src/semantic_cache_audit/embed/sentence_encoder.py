"""A real pretrained sentence encoder, run through ONNX Runtime.

This is the embedder the README quotes. It is a genuine, widely deployed model rather
than anything invented here, and that matters: the central finding of this repository
is a statement about what a production semantic cache does, and it is only worth
making about a model teams actually deploy.

Why ONNX Runtime and a tokenizer rather than the sentence-transformers package: the
package pulls a deep learning framework that this repository otherwise has no use
for, and the pooling it applies to this model is three lines. Running the graph
directly keeps the dependency surface honest about what is actually being used.

Nothing on the default path ever constructs this class. `make verify` reads the
committed matrix in `fixtures/embeddings/`, which is why a reviewer with no weights
and no network reproduces every published number. Weights are fetched only by
`make embeddings-live`, and ADR-002 records the split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Embedder, l2_normalise

#: The model every headline figure in this repository is a fact about.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#: Files the encoder needs. The ONNX graph is the weights; the rest is tokenisation.
REQUIRED_FILES = ("tokenizer.json", "onnx/model.onnx")

MAX_TOKENS = 256


class EncoderUnavailable(RuntimeError):
    """The optional encoder extra is not installed, or the weights are not present."""


class OnnxSentenceEncoder(Embedder):
    """Mean pooled, L2 normalised sentence embeddings from a transformer encoder.

    Mean pooling over the attention mask, then normalisation, is the pooling this
    model card specifies. Getting it wrong is silent: the vectors still look like
    vectors and every similarity is subtly off, so the pooling is pinned by a test
    that checks a known pair of sentences against the published behaviour.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, local_dir: str | Path | None = None):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise EncoderUnavailable(
                "the encoder extra is not installed. Run: pip install -e '.[encoder]'. "
                "This is only needed to regenerate the embedding fixture; `make verify` "
                "reads the committed matrix and needs neither weights nor network."
            ) from exc

        self._model_name = model_name
        paths = self._resolve(model_name, local_dir)
        self._tokenizer = Tokenizer.from_file(str(paths["tokenizer.json"]))
        self._tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self._tokenizer.enable_padding()
        self._session = ort.InferenceSession(
            str(paths["onnx/model.onnx"]), providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._dim = int(self._session.get_outputs()[0].shape[-1])

    @staticmethod
    def _resolve(model_name: str, local_dir: str | Path | None) -> dict[str, Path]:
        if local_dir is not None:
            root = Path(local_dir)
            paths = {f: root / f for f in REQUIRED_FILES}
            missing = [f for f, p in paths.items() if not p.exists()]
            if missing:
                raise EncoderUnavailable(f"{root} is missing {missing}")
            return paths
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise EncoderUnavailable(
                "huggingface_hub is not installed and no local_dir was given"
            ) from exc
        return {f: Path(hf_hub_download(repo_id=model_name, filename=f)) for f in REQUIRED_FILES}

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"st:{self._model_name}"

    @property
    def dim(self) -> int:
        return self._dim

    def describe(self) -> dict:
        return {
            "name": self.name,
            "dim": self._dim,
            "kind": type(self).__name__,
            "model": self._model_name,
            "pooling": "mean over attention mask, then L2 normalise",
            "max_tokens": MAX_TOKENS,
            "pretrained": True,
        }

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        out = [self._encode_batch(texts[i : i + batch_size])
               for i in range(0, len(texts), batch_size)]
        return l2_normalise(np.vstack(out).astype(np.float32))

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encoded = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array([e.type_ids for e in encoded], dtype=np.int64)
        hidden = self._session.run(None, feed)[0]
        weights = mask[..., None].astype(np.float32)
        return (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
