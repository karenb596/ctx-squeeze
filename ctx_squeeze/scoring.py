"""TF-IDF style extractive scoring for segments.

Every other selection strategy keeps a contiguous slice of the document (a
head, a tail, whatever survives dedupe). This one instead ranks segments by
how much they say that nothing else in the document says: a paragraph built
from words that recur everywhere scores low, since dropping it loses little;
a paragraph carrying words found nowhere else scores high, on the theory that
generic connective text is safe to cut and distinctive detail is not.
"""

import math
import re

__all__ = ["score_segments", "select_by_score"]

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")

_STOPWORDS = frozenset(
    """
    the a an and or but if then else of to in on for with as by at is are
    was were be been being this that these those it its from into than so
    not no do does did has have had will would can could should may might
    you your we our i he she they them his her their there here which who
    what when how
    """.split()
)


def _words(text):
    return [w.lower() for w in _WORD_RE.findall(text) if len(w) > 2]


def score_segments(segments):
    """Return one relevance score per segment, aligned with ``segments``.

    Each score is the average tf-idf weight of the segment's non-stopword
    words: term frequency within the segment times inverse document
    frequency across all segments, divided by word count so a long segment
    can't win purely by repeating a rare word many times. A segment with no
    scorable words gets 0.0.
    """
    per_segment_words = [
        [w for w in _words(segment.text) if w not in _STOPWORDS] for segment in segments
    ]

    doc_freq = {}
    for words in per_segment_words:
        for word in set(words):
            doc_freq[word] = doc_freq.get(word, 0) + 1

    n_docs = len(segments)
    scores = []
    for words in per_segment_words:
        if not words:
            scores.append(0.0)
            continue
        term_freq = {}
        for word in words:
            term_freq[word] = term_freq.get(word, 0) + 1
        total = 0.0
        for word, tf in term_freq.items():
            idf = math.log((n_docs + 1) / (doc_freq[word] + 1)) + 1.0
            total += tf * idf
        scores.append(total / len(words))
    return scores


def select_by_score(segments, budget, scores=None):
    """Greedily keep the highest-scoring segments that fit in ``budget`` tokens.

    Segments are offered to the budget in score order, highest first, so a
    later, cheaper segment can still fit after an earlier, pricier one is
    skipped. Ties fall back to document order for a deterministic result.
    The return value preserves the original document order of the kept
    segments, not the selection order, so callers can re-join them directly.

    Returns ``(kept, dropped_count)``.
    """
    if scores is None:
        scores = score_segments(segments)

    order = sorted(range(len(segments)), key=lambda i: (-scores[i], i))

    kept_indices = set()
    used = 0
    for i in order:
        cost = segments[i].tokens
        if used + cost > budget:
            continue
        kept_indices.add(i)
        used += cost

    kept = [segments[i] for i in range(len(segments)) if i in kept_indices]
    return kept, len(segments) - len(kept)
