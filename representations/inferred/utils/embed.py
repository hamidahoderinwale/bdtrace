"""Embed text for similarity comparison."""

_EMBEDDER = None
_DEFAULT_MODEL = "all-MiniLM-L6-v2"


def embed_text(text: str, model: str | None = None) -> list[float]:
    """Embed text for similarity comparison."""
    if not text or not text.strip():
        return []
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(model or _DEFAULT_MODEL)
    emb = _EMBEDDER.encode(text.strip(), convert_to_numpy=True)
    return emb.tolist()
