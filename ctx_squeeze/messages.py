"""Structural pruning for chat transcripts.

Unlike the document compactor, a chat transcript can't be cut at an arbitrary
point: a tool result that shows up without the call that produced it (or the
reverse) is a malformed transcript most providers will reject outright, and a
system prompt or the last few turns of context are usually the reason the
model still understands what it's doing. So pruning here works on whole
messages, keeps a protected window intact regardless of budget, and treats a
tool call and its result as a single unit that lives or dies together no
matter how far apart they land in the transcript.
"""

import json

from .tokens import estimate_tokens

__all__ = ["Message", "PruneResult", "parse_messages", "prune_messages"]


class Message(object):
    """One parsed transcript message.

    ``text`` is a flattened, human-readable view used for token estimation;
    it doesn't round-trip. ``raw`` is the original dict, untouched, and is
    what actually gets emitted.
    """

    __slots__ = ("raw", "role", "text", "tokens", "tool_call_ids", "tool_result_ids", "index")

    def __init__(self, raw, role, text, tool_call_ids, tool_result_ids, index):
        self.raw = raw
        self.role = role
        self.text = text
        self.tokens = estimate_tokens(text)
        self.tool_call_ids = tool_call_ids
        self.tool_result_ids = tool_result_ids
        self.index = index

    def to_dict(self):
        return self.raw

    def __repr__(self):
        return "Message(index=%d, role=%s, tokens=%d)" % (self.index, self.role, self.tokens)


class PruneResult(object):
    """Result of :func:`prune_messages`."""

    __slots__ = (
        "messages",
        "pinned_tool_results",
        "original_tokens",
        "final_tokens",
        "messages_in",
        "messages_out",
        "notes",
    )

    def __init__(
        self,
        messages,
        pinned_tool_results,
        original_tokens,
        final_tokens,
        messages_in,
        messages_out,
        notes,
    ):
        self.messages = messages
        self.pinned_tool_results = pinned_tool_results
        self.original_tokens = original_tokens
        self.final_tokens = final_tokens
        self.messages_in = messages_in
        self.messages_out = messages_out
        self.notes = notes

    def to_dicts(self):
        """Return the kept messages as plain dicts, ready for ``json.dumps``."""
        return [message.raw for message in self.messages]

    def __repr__(self):
        return "PruneResult(messages_out=%d of %d, tokens=%d -> %d)" % (
            self.messages_out,
            self.messages_in,
            self.original_tokens,
            self.final_tokens,
        )


def _text_from_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                parts.append(block.get("text", ""))
            elif block_type == "tool_result":
                inner = block.get("content")
                if isinstance(inner, str):
                    parts.append(inner)
                elif isinstance(inner, list):
                    parts.append(_text_from_content(inner))
        return "\n".join(p for p in parts if p)
    return ""


def _flatten_text(raw, content):
    """Join every human-readable piece of a message into one string.

    Tool call arguments count toward the token estimate too: a message whose
    only content is a large JSON tool call is not free just because its
    ``content`` field is empty.
    """
    parts = [_text_from_content(content)]
    for call in raw.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        arguments = function.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        parts.append("%s(%s)" % (function.get("name", ""), arguments))
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                arguments = block.get("input", "")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                parts.append("%s(%s)" % (block.get("name", ""), arguments))
    return "\n".join(p for p in parts if p)


def _tool_call_ids(raw, content):
    ids = []
    for call in raw.get("tool_calls") or []:
        if isinstance(call, dict) and call.get("id"):
            ids.append(call["id"])
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id"):
                ids.append(block["id"])
    return ids


def _tool_result_ids(raw, content):
    ids = []
    if raw.get("tool_call_id"):
        ids.append(raw["tool_call_id"])
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id"):
                ids.append(block["tool_use_id"])
    return ids


def parse_messages(history):
    """Parse a list of raw chat message dicts into :class:`Message` objects.

    Accepts the OpenAI chat shape (string ``content``, tool calls under
    ``tool_calls`` / ``tool_call_id``) and the Anthropic shape (``content``
    a list of typed blocks, tool calls as ``tool_use`` / ``tool_result``
    blocks). Both can appear in the same list, since a transcript replayed
    through more than one provider ends up with a mix.
    """
    messages = []
    for i, raw in enumerate(history):
        content = raw.get("content")
        messages.append(
            Message(
                raw=raw,
                role=raw.get("role", "user"),
                text=_flatten_text(raw, content),
                tool_call_ids=_tool_call_ids(raw, content),
                tool_result_ids=_tool_result_ids(raw, content),
                index=i,
            )
        )
    return messages


def _group_turns(messages):
    """Group non-system messages so each turn starts at a user message.

    Any assistant or tool messages before the first user message form a
    leading turn of their own, so they stay eligible for pinning or
    dropping like any other turn instead of silently surviving.
    """
    turns = []
    current = []
    for message in messages:
        if message.role == "user" and current:
            turns.append(current)
            current = []
        current.append(message)
    if current:
        turns.append(current)
    return turns


def _build_groups(messages):
    """Union-find messages that share a tool call id, in either direction.

    A group is the smallest set of messages that all have to be kept or
    dropped together: the assistant message that issued a tool call and the
    message that answers it, transitively, since one assistant turn can
    issue several calls answered by several separate messages.
    """
    parent = list(range(len(messages)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    id_to_indices = {}
    for message in messages:
        for tid in message.tool_call_ids:
            id_to_indices.setdefault(tid, []).append(message.index)
        for tid in message.tool_result_ids:
            id_to_indices.setdefault(tid, []).append(message.index)
    for indices in id_to_indices.values():
        for other in indices[1:]:
            union(indices[0], other)

    roots = [find(i) for i in range(len(messages))]
    groups = {}
    for i, root in enumerate(roots):
        groups.setdefault(root, []).append(i)
    return roots, groups


def _marker_message(count):
    text = "[%d earlier messages elided]" % count
    raw = {"role": "system", "content": text}
    return Message(raw=raw, role="system", text=text, tool_call_ids=[], tool_result_ids=[], index=-1)


def prune_messages(messages, budget, recent_turns=2, marker=True):
    """Prune a parsed transcript to fit ``budget`` estimated tokens.

    System messages and the most recent ``recent_turns`` user turns are kept
    whole no matter the budget. Everything else is a candidate for removal,
    offered back in most-recent-first order until nothing more fits (a later,
    cheaper message can still make it in after an earlier, pricier one is
    skipped). A tool call and the result that answers it are always kept or
    dropped together, even when one of the two already falls inside the
    protected window and the other doesn't.

    ``messages`` must be in original chronological order, as returned by
    :func:`parse_messages`.

    Returns a :class:`PruneResult`.
    """
    if not messages:
        return PruneResult(
            messages=[],
            pinned_tool_results=[],
            original_tokens=0,
            final_tokens=0,
            messages_in=0,
            messages_out=0,
            notes=[],
        )

    system_indices = {m.index for m in messages if m.role == "system"}
    turns = _group_turns([m for m in messages if m.role != "system"])
    recent = turns[-recent_turns:] if recent_turns > 0 else []
    recent_indices = {m.index for turn in recent for m in turn}

    roots, groups = _build_groups(messages)

    mandatory = set()
    for idx in system_indices | recent_indices:
        mandatory.update(groups[roots[idx]])

    used = sum(messages[i].tokens for i in mandatory)
    remaining = budget - used

    items = {}
    for message in messages:
        if message.index in mandatory:
            continue
        items.setdefault(roots[message.index], []).append(message.index)
    item_list = sorted(items.values(), key=max, reverse=True)

    kept = set(mandatory)
    added = 0
    for item in item_list:
        cost = sum(messages[i].tokens for i in item)
        if cost <= remaining:
            kept.update(item)
            remaining -= cost
            added += len(item)

    notes = []
    if added:
        notes.append("kept %d older message(s) outside the recent window" % added)

    pinned_tool_results = set()
    for idx in kept:
        if idx in system_indices or idx in recent_indices:
            continue
        pinned_tool_results.update(messages[idx].tool_call_ids)
        pinned_tool_results.update(messages[idx].tool_result_ids)
    if pinned_tool_results:
        notes.append("pinned %d tool result(s) linked across the recent window" % len(pinned_tool_results))

    total_dropped = len(messages) - len(kept)
    if total_dropped:
        notes.append("dropped %d message(s) outside the kept window" % total_dropped)

    output = []
    pending = 0
    for message in messages:
        if message.index in kept:
            if pending and marker:
                output.append(_marker_message(pending))
            pending = 0
            output.append(message)
        else:
            pending += 1
    if pending and marker:
        output.append(_marker_message(pending))

    return PruneResult(
        messages=output,
        pinned_tool_results=sorted(pinned_tool_results),
        original_tokens=sum(m.tokens for m in messages),
        final_tokens=sum(m.tokens for m in output),
        messages_in=len(messages),
        messages_out=len(output),
        notes=notes,
    )
