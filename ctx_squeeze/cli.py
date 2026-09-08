"""Command-line entry point for ctx-squeeze.

The CLI is a thin wrapper: it turns argv into keyword arguments for
``squeeze`` or ``prune_messages`` and formats the result. Anything that
looks like a decision (how to pick segments, how to pin tool results)
belongs in the library, not here.
"""

import argparse
import json
import sys

from .compactor import STRATEGIES, squeeze
from .messages import parse_messages, prune_messages

__all__ = ["main", "build_parser"]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ctx-squeeze",
        description="Fit a document or chat transcript into an estimated token budget.",
    )
    parser.add_argument("input", help="path to the input file, or - to read stdin")
    parser.add_argument("--budget", type=int, required=True, help="target size in estimated tokens")
    parser.add_argument(
        "--strategy",
        default="score",
        help="comma-separated pipeline: %s (default: score)" % ", ".join(sorted(STRATEGIES)),
    )
    parser.add_argument(
        "--head-ratio", type=float, default=0.5, help="share of the budget spent on the head in head-tail"
    )
    parser.add_argument(
        "--jaccard", type=float, default=0.8, help="similarity at which two segments count as duplicates"
    )
    parser.add_argument("--shingle-size", type=int, default=5, help="words per shingle in the dedupe stage")
    parser.add_argument("--messages", action="store_true", help="treat the input as a JSON chat transcript")
    parser.add_argument("--recent-turns", type=int, default=2, help="user turns kept whole in --messages mode")
    parser.add_argument("--no-marker", action="store_true", help="omit the [N segments elided] markers")
    parser.add_argument("--stats", action="store_true", help="print a token summary to stderr")
    parser.add_argument("--json", action="store_true", help="emit a JSON report instead of plain text")
    parser.add_argument("-o", metavar="PATH", dest="output", help="write the result to a file (default: stdout)")
    return parser


def _read_input(path):
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _write_output(path, text):
    if path is None:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _run_document(text, args):
    result = squeeze(
        text,
        budget=args.budget,
        strategy=args.strategy,
        head_ratio=args.head_ratio,
        jaccard=args.jaccard,
        shingle_size=args.shingle_size,
        marker=not args.no_marker,
    )
    stats_line = "kept %d of %d segments | %d -> %d tokens (budget %d)" % (
        result.segments_out,
        result.segments_in,
        result.original_tokens,
        result.final_tokens,
        args.budget,
    )
    report = {
        "text": result.text,
        "original_tokens": result.original_tokens,
        "final_tokens": result.final_tokens,
        "segments_in": result.segments_in,
        "segments_out": result.segments_out,
        "notes": result.notes,
    }
    return result.text, stats_line, report


def _run_messages(text, args):
    history = json.loads(text)
    if not isinstance(history, list):
        raise ValueError("expected a JSON array of messages")
    messages = parse_messages(history)
    result = prune_messages(
        messages,
        budget=args.budget,
        recent_turns=args.recent_turns,
        marker=not args.no_marker,
    )
    output_text = json.dumps(result.to_dicts(), indent=2)
    stats_line = "kept %d of %d messages | %d -> %d tokens (budget %d)" % (
        result.messages_out,
        result.messages_in,
        result.original_tokens,
        result.final_tokens,
        args.budget,
    )
    report = {
        "text": output_text,
        "original_tokens": result.original_tokens,
        "final_tokens": result.final_tokens,
        "messages_in": result.messages_in,
        "messages_out": result.messages_out,
        "pinned_tool_results": result.pinned_tool_results,
        "notes": result.notes,
    }
    return output_text, stats_line, report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        text = _read_input(args.input)
    except OSError as exc:
        parser.error(str(exc))

    try:
        if args.messages:
            output_text, stats_line, report = _run_messages(text, args)
        else:
            output_text, stats_line, report = _run_document(text, args)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    if args.stats:
        sys.stderr.write(stats_line + "\n")

    if args.json:
        _write_output(args.output, json.dumps(report, indent=2))
    else:
        _write_output(args.output, output_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
