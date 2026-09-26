from ctx_squeeze.dedupe import dedupe_segments, find_near_duplicates, jaccard, shingles
from ctx_squeeze.segments import Segment


def make_segment(text, kind="text", index=0):
    return Segment(text, start_line=1, end_line=1, kind=kind, index=index)


def test_shingles_of_empty_text_is_empty_set():
    assert shingles("") == set()
    assert shingles("   ") == set()


def test_shingles_of_short_text_is_single_shingle():
    result = shingles("one two three", size=5)
    assert result == {("one", "two", "three")}


def test_shingles_lowercases_and_splits_on_whitespace():
    result = shingles("One  Two", size=2)
    assert result == {("one", "two")}


def test_shingles_slides_a_window_over_longer_text():
    result = shingles("a b c d e f", size=5)
    assert result == {
        ("a", "b", "c", "d", "e"),
        ("b", "c", "d", "e", "f"),
    }


def test_jaccard_of_two_empty_sets_is_one():
    assert jaccard(set(), set()) == 1.0


def test_jaccard_of_one_empty_set_is_zero():
    assert jaccard({("a",)}, set()) == 0.0
    assert jaccard(set(), {("a",)}) == 0.0


def test_jaccard_of_identical_sets_is_one():
    a = {("a", "b"), ("b", "c")}
    assert jaccard(a, a) == 1.0


def test_jaccard_is_intersection_over_union():
    a = {("a",), ("b",), ("c",)}
    b = {("b",), ("c",), ("d",)}
    assert jaccard(a, b) == 2 / 4


def test_find_near_duplicates_flags_exact_repeat():
    segments = [
        make_segment("the quick brown fox jumps over", index=0),
        make_segment("the quick brown fox jumps over", index=1),
    ]
    assert find_near_duplicates(segments) == [1]


def test_find_near_duplicates_flags_near_match_with_one_changed_word():
    # 9 of 10 words shared, one byte count differs: jaccard 9/11 with
    # single-word shingles, comfortably above the default 0.8 threshold.
    segments = [
        make_segment("reading file config.py got 4096 bytes total after first attempt", index=0),
        make_segment("reading file config.py got 8192 bytes total after first attempt", index=1),
    ]
    assert find_near_duplicates(segments, shingle_size=1) == [1]


def test_find_near_duplicates_ignores_unrelated_text():
    segments = [
        make_segment("the quick brown fox jumps over", index=0),
        make_segment("something completely different here", index=1),
    ]
    assert find_near_duplicates(segments) == []


def test_find_near_duplicates_compares_against_all_survivors_not_just_previous():
    segments = [
        make_segment("the quick brown fox jumps over", index=0),
        make_segment("an unrelated line in between", index=1),
        make_segment("the quick brown fox jumps over", index=2),
    ]
    assert find_near_duplicates(segments) == [2]


def test_find_near_duplicates_respects_kind():
    segments = [
        make_segment("def f(): return 1 + 2 + 3", kind="code", index=0),
        make_segment("def f(): return 1 + 2 + 3", kind="text", index=1),
    ]
    assert find_near_duplicates(segments) == []


def test_dedupe_segments_drops_repeats_and_keeps_first_occurrence():
    segments = [
        make_segment("the quick brown fox jumps over", index=0),
        make_segment("the quick brown fox jumps over", index=1),
        make_segment("a third and different segment", index=2),
    ]
    kept, dropped = dedupe_segments(segments)
    assert dropped == 1
    assert [s.index for s in kept] == [0, 2]


def test_dedupe_segments_with_no_duplicates_keeps_everything():
    segments = [
        make_segment("first unique segment here", index=0),
        make_segment("second unique segment here", index=1),
    ]
    kept, dropped = dedupe_segments(segments)
    assert dropped == 0
    assert kept == segments


def test_dedupe_segments_of_empty_list():
    kept, dropped = dedupe_segments([])
    assert kept == []
    assert dropped == 0
