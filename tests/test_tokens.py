import math

from ctx_squeeze.tokens import (
    CHARS_PER_NUMBER_TOKEN,
    CHARS_PER_WORD_TOKEN,
    TOKENS_PER_CJK_CHAR,
    TOKENS_PER_NEWLINE,
    TOKENS_PER_SYMBOL,
    estimate_tokens,
    fits_budget,
    truncate_to_tokens,
)


def test_empty_and_whitespace_only_are_zero():
    assert estimate_tokens("") == 0
    assert estimate_tokens("   \t  ") == 0


def test_short_word_costs_one_token():
    # below CHARS_PER_WORD_TOKEN, so the per-word minimum kicks in
    assert estimate_tokens("cat") == 1


def test_long_word_scales_with_length():
    word = "a" * 40
    expected = math.ceil(len(word) / CHARS_PER_WORD_TOKEN)
    assert estimate_tokens(word) == expected


def test_digit_run_uses_number_rate():
    digits = "123456789"
    expected = math.ceil(len(digits) / CHARS_PER_NUMBER_TOKEN)
    assert estimate_tokens(digits) == expected


def test_short_digit_run_costs_one_token():
    assert estimate_tokens("42") == 1


def test_cjk_is_one_token_per_character():
    text = "一二三"  # three CJK ideographs
    assert estimate_tokens(text) == math.ceil(len(text) * TOKENS_PER_CJK_CHAR)


def test_newlines_are_cheaper_than_words():
    assert estimate_tokens("\n\n\n") == math.ceil(3 * TOKENS_PER_NEWLINE)


def test_symbols_use_symbol_rate():
    assert estimate_tokens("!!!") == math.ceil(3 * TOKENS_PER_SYMBOL)


def test_extra_spaces_between_words_do_not_add_tokens():
    # the space run itself is free regardless of how long it is
    assert estimate_tokens("hello world") == estimate_tokens("hello     world")


def test_apostrophe_word_is_a_single_token_run():
    assert estimate_tokens("don't") == max(1, math.ceil(len("don't") / CHARS_PER_WORD_TOKEN))


def test_result_is_never_negative():
    assert estimate_tokens("x") >= 0


def test_fits_budget_true_and_false():
    text = "a" * 100
    tokens = estimate_tokens(text)
    assert fits_budget(text, tokens) is True
    assert fits_budget(text, tokens - 1) is False


def test_truncate_returns_unchanged_text_when_it_already_fits():
    text = "short sentence"
    assert truncate_to_tokens(text, 1000) == text


def test_truncate_with_zero_or_negative_budget_is_empty():
    assert truncate_to_tokens("anything", 0) == ""
    assert truncate_to_tokens("anything", -5) == ""


def test_truncate_shrinks_to_fit_budget():
    text = "word " * 200
    budget = 10
    result = truncate_to_tokens(text, budget)
    assert estimate_tokens(result) <= budget
    assert len(result) < len(text)


def test_truncate_appends_suffix_and_still_fits():
    text = "word " * 200
    budget = 10
    suffix = " [cut]"
    result = truncate_to_tokens(text, budget, suffix=suffix)
    assert result.endswith(suffix)
    assert estimate_tokens(result) <= budget


def test_truncate_returns_empty_when_suffix_alone_exceeds_budget():
    suffix = "a" * 400  # far more tokens than the budget allows on its own
    result = truncate_to_tokens("word " * 200, budget=1, suffix=suffix)
    assert result == ""


def test_truncate_strips_trailing_whitespace_before_suffix():
    text = "one two three four five six seven eight nine ten"
    result = truncate_to_tokens(text, budget=3, suffix="...")
    body = result[: -len("...")] if result.endswith("...") else result
    assert body == body.rstrip()
