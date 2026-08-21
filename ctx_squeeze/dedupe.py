"""Near-duplicate detection for segments, using shingle overlap.

Agent transcripts repeat text almost verbatim except for a detail that keeps
it from matching exactly: the same file read three times with a different
byte count, the same traceback after each retry with a different timestamp.
Word n-gram shingles plus Jaccard similarity catch those near-matches without
diffing or embeddings, and stay cheap enough to run on every segment pair in
a long transcript.
"""

import re

__all__ = ["shingles", "jaccard", "find_near_duplicates", "dedupe_segments"]

_TOKEN_RE = re.compile(r"\S+")


def shingles(text, size=5):
    """Return the set of ``size``-word shingles in ``text``.

    Shingles are built from whitespace-separated tokens, lowercased so that
    punctuation and casing differences don't block a match. A text with
    fewer than ``size`` words yields a single shingle covering all of it,
    rather than an empty set, so short segments can still be compared.
    """
    words = _TOKEN_RE.findall(text.lower())
    if not words:
        return set()
    if len(words) <= size:
        return {tuple(words)}
    return set(tuple(words[i : i + size]) for i in range(len(words) - size + 1))


def jaccard(a, b):
    """Jaccard similarity between two shingle sets: |intersection| / |union|.

    Two empty sets are treated as identical (similarity 1.0) rather than
    undefined, since two segments with nothing to shingle are the same kind
    of nothing.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union


def find_near_duplicates(segments, threshold=0.8, shingle_size=5):
    """Return the indices of segments that repeat an earlier segment.

    Every candidate is compared against every segment kept so far, not just
    its immediate predecessor, since a repeated block can resurface several
    turns later with unrelated text in between. A segment is only compared
    against survivors of the same ``kind``: a short code diff and a short
    log line can share plenty of shingles by accident, but they aren't the
    same content repeating.
    """
    survivors = []  # (kind, shingle set) for segments not yet marked duplicate
    duplicates = []
    for i, segment in enumerate(segments):
        candidate = shingles(segment.text, shingle_size)
        is_duplicate = False
        for kind, kept_shingles in survivors:
            if kind == segment.kind and jaccard(candidate, kept_shingles) >= threshold:
                is_duplicate = True
                break
        if is_duplicate:
            duplicates.append(i)
        else:
            survivors.append((segment.kind, candidate))
    return duplicates


def dedupe_segments(segments, threshold=0.8, shingle_size=5):
    """Drop near-duplicate segments, keeping the first occurrence of each.

    Returns ``(kept, dropped_count)``. ``kept`` preserves document order and
    each surviving :class:`~ctx_squeeze.segments.Segment` keeps its original
    ``index``, so anything built on top can still tell where a gap in the
    numbering came from a dropped duplicate rather than a budget cut.
    """
    drop = set(find_near_duplicates(segments, threshold, shingle_size))
    kept = [segment for i, segment in enumerate(segments) if i not in drop]
    return kept, len(drop)
