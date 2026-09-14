from ctx_squeeze.segments import Segment, join_segments, split_segments


def test_empty_and_whitespace_only_text_yields_no_segments():
    assert split_segments("") == []
    assert split_segments("   \n\t  \n") == []


def test_single_paragraph_is_one_segment():
    segments = split_segments("just one paragraph")
    assert len(segments) == 1
    assert segments[0].text == "just one paragraph"
    assert segments[0].kind == "text"


def test_blank_lines_separate_paragraphs():
    text = "first paragraph\n\nsecond paragraph"
    segments = split_segments(text)
    assert [s.text for s in segments] == ["first paragraph", "second paragraph"]


def test_multiple_blank_lines_collapse_to_one_split():
    text = "first\n\n\n\nsecond"
    segments = split_segments(text)
    assert [s.text for s in segments] == ["first", "second"]


def test_multiline_paragraph_stays_one_segment():
    text = "line one\nline two\nline three"
    segments = split_segments(text)
    assert len(segments) == 1
    assert segments[0].text == text


def test_fenced_code_block_is_kept_whole_and_marked_code():
    text = "before\n\n```\ncode line one\n\ncode line two\n```\n\nafter"
    segments = split_segments(text)
    kinds = [s.kind for s in segments]
    assert kinds == ["text", "code", "text"]
    assert segments[1].text == "```\ncode line one\n\ncode line two\n```"
    assert segments[1].is_code is True


def test_tilde_fence_is_also_treated_as_code():
    text = "~~~\nsome code\n~~~"
    segments = split_segments(text)
    assert len(segments) == 1
    assert segments[0].kind == "code"


def test_unterminated_fence_runs_to_end_of_input():
    text = "intro\n\n```\ndangling code\nstill inside"
    segments = split_segments(text)
    assert [s.kind for s in segments] == ["text", "code"]
    assert segments[1].text == "```\ndangling code\nstill inside"


def test_segments_are_indexed_in_document_order():
    text = "one\n\ntwo\n\nthree"
    segments = split_segments(text)
    assert [s.index for s in segments] == [0, 1, 2]


def test_start_and_end_line_are_one_based_and_span_the_segment():
    text = "para one\n\npara two line a\npara two line b"
    segments = split_segments(text)
    assert segments[0].start_line == 1
    assert segments[0].end_line == 1
    assert segments[1].start_line == 3
    assert segments[1].end_line == 4


def test_tokens_are_computed_on_construction():
    segment = Segment("hello world", start_line=1, end_line=1)
    assert segment.tokens > 0


def test_join_segments_uses_separator_between_chunks():
    text = "alpha\n\nbeta\n\ngamma"
    segments = split_segments(text)
    assert join_segments(segments) == "alpha\n\nbeta\n\ngamma"
    assert join_segments(segments, separator=" | ") == "alpha | beta | gamma"


def test_join_segments_of_empty_list_is_empty_string():
    assert join_segments([]) == ""


def test_segment_equality_compares_all_fields():
    a = Segment("text", start_line=1, end_line=2, kind="text", index=0)
    b = Segment("text", start_line=1, end_line=2, kind="text", index=0)
    c = Segment("text", start_line=1, end_line=2, kind="code", index=0)
    assert a == b
    assert a != c
    assert a != "not a segment"


def test_segment_repr_includes_key_fields():
    segment = Segment("abc", start_line=1, end_line=1, kind="text", index=2)
    text = repr(segment)
    assert "index=2" in text
    assert "kind=text" in text
    assert "lines=1-1" in text
