"""Top-level pipeline that turns a strategy name into a budget-fitting document.

The three segment-level tools (dedupe, score, head-tail) each know how to trim
a list of segments; none of them alone is "the" compactor. ``squeeze`` is what
chains them, in whatever order the caller asks for, and is the one place that
promises the output actually fits the budget - the individual stages are
heuristics and can leave the joined text a little over, once separators and
elision markers are added back in, so a hard truncation pass at the end is
what makes the promise true rather than usually true.
"""

from .dedupe import dedupe_segments
from .scoring import select_by_score
from .segments import split_segments
from .tokens import estimate_tokens, truncate_to_tokens

__all__ = ["STRATEGIES", "SqueezeResult", "squeeze"]


class SqueezeResult(object):
    """Result of :func:`squeeze`."""

    __slots__ = (
        "text",
        "original_tokens",
        "final_tokens",
        "segments_in",
        "segments_out",
        "notes",
    )

    def __init__(self, text, original_tokens, final_tokens, segments_in, segments_out, notes):
        self.text = text
        self.original_tokens = original_tokens
        self.final_tokens = final_tokens
        self.segments_in = segments_in
        self.segments_out = segments_out
        self.notes = notes

    def __repr__(self):
        return "SqueezeResult(segments_out=%d of %d, tokens=%d -> %d)" % (
            self.segments_out,
            self.segments_in,
            self.original_tokens,
            self.final_tokens,
        )


def _stage_dedupe(segments, budget, notes, **options):
    kept, dropped = dedupe_segments(
        segments,
        threshold=options.get("jaccard", 0.8),
        shingle_size=options.get("shingle_size", 5),
    )
    if dropped:
        notes.append("dedupe dropped %d near-duplicate segment(s)" % dropped)
    return kept


def _stage_score(segments, budget, notes, **options):
    kept, dropped = select_by_score(segments, budget)
    if dropped:
        notes.append("score dropped %d segment(s) outside the budget" % dropped)
    return kept


def _stage_head_tail(segments, budget, notes, **options):
    head_ratio = options.get("head_ratio", 0.5)
    head_budget = int(round(budget * head_ratio))
    tail_budget = max(budget - head_budget, 0)

    head_kept = []
    used = 0
    for segment in segments:
        if used + segment.tokens > head_budget:
            break
        head_kept.append(segment)
        used += segment.tokens

    tail_candidates = segments[len(head_kept):]
    tail_kept = []
    used = 0
    for segment in reversed(tail_candidates):
        if used + segment.tokens > tail_budget:
            break
        tail_kept.append(segment)
        used += segment.tokens
    tail_kept.reverse()

    kept = head_kept + tail_kept
    dropped = len(segments) - len(kept)
    if dropped:
        notes.append("head-tail dropped %d segment(s) outside the budget" % dropped)
    return kept


STRATEGIES = {
    "dedupe": _stage_dedupe,
    "score": _stage_score,
    "head-tail": _stage_head_tail,
}


def _build_output(all_segments, kept_segments, marker):
    """Re-join the original segments, collapsing runs of dropped ones.

    Walking ``all_segments`` rather than ``kept_segments`` is what lets a gap
    left by an earlier stage (dedupe cutting from the middle) and a gap left
    by a later one (score cutting from wherever) collapse into a single
    marker instead of one marker per stage.
    """
    kept_ids = set(id(segment) for segment in kept_segments)
    parts = []
    pending = 0
    for segment in all_segments:
        if id(segment) in kept_ids:
            if pending and marker:
                parts.append("[%d segments elided]" % pending)
            pending = 0
            parts.append(segment.text)
        else:
            pending += 1
    if pending and marker:
        parts.append("[%d segments elided]" % pending)
    return "\n\n".join(parts)


def squeeze(text, budget, strategy="score", head_ratio=0.5, jaccard=0.8, shingle_size=5, marker=True):
    """Compact ``text`` to fit ``budget`` estimated tokens.

    ``strategy`` is a comma-separated list of stage names from
    :data:`STRATEGIES`, applied left to right; each stage receives the
    previous stage's surviving segments. ``head_ratio``, ``jaccard`` and
    ``shingle_size`` are passed to whichever stages use them and ignored by
    the rest.

    Returns a :class:`SqueezeResult`. ``result.final_tokens`` is never above
    ``budget``: if the stages leave the joined text (markers included) over
    budget, the tail is hard-truncated as a last resort.
    """
    all_segments = split_segments(text)
    original_tokens = estimate_tokens(text)
    notes = []

    segments = all_segments
    for name in (part.strip() for part in strategy.split(",")):
        if not name:
            continue
        stage = STRATEGIES.get(name)
        if stage is None:
            raise ValueError(
                "unknown strategy %r (choose from %s)" % (name, ", ".join(sorted(STRATEGIES)))
            )
        segments = stage(
            segments,
            budget,
            notes,
            head_ratio=head_ratio,
            jaccard=jaccard,
            shingle_size=shingle_size,
        )

    output_text = _build_output(all_segments, segments, marker)
    final_tokens = estimate_tokens(output_text)
    if final_tokens > budget:
        output_text = truncate_to_tokens(output_text, budget)
        final_tokens = estimate_tokens(output_text)
        notes.append("hard-truncated output to stay within budget")

    return SqueezeResult(
        text=output_text,
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        segments_in=len(all_segments),
        segments_out=len(segments),
        notes=notes,
    )
