"""Text preprocessing with spaCy.

Loads the small English model, tokenizes, lemmatizes, and strips
stop words / punctuation / whitespace tokens.  The spaCy pipeline is
loaded once (module-level lazy singleton) and reused across calls.
"""

from __future__ import annotations

import functools

import spacy

# The spaCy model we download once (`python -m spacy download en_core_web_sm`).
MODEL_NAME = "en_core_web_sm"


# loading spaCy is slow, so this loads it exactly once and hands the
# same copy to every later call (what lru_cache does). "exclude"
# skips the parts we never use (named entities, sentence parsing)
@functools.lru_cache(maxsize=1)
def get_nlp() -> spacy.Language:
    """Load (and cache) the spaCy English pipeline."""
    try:
        return spacy.load(MODEL_NAME, exclude=["ner", "parser"])
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{MODEL_NAME}' is not installed. "
            f"Run: python -m spacy download {MODEL_NAME}"
        ) from exc


# shrink a sentence to useful content words
def preprocess(text: str) -> str:
    """Return a cleaned, lemmatized, space-joined token string."""
    if not text or not text.strip():
        return ""
    doc = get_nlp()(text)
    keep = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
        and token.lemma_.strip()
    ]
    return " ".join(keep)


# same cleaning as above but for a whole list, using spaCy's batched pipe() which is much faster
def preprocess_many(texts: list[str]) -> list[str]:
    """Preprocess a batch of texts using nlp.pipe for speed."""
    texts = [t if t and t.strip() else "" for t in texts]
    docs = get_nlp().pipe(texts, batch_size=64)
    return [
        " ".join(
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
            and token.lemma_.strip()
        )
        for doc in docs
    ]
