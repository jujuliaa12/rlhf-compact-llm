"""Unit tests for src.preprocessing."""
from __future__ import annotations

from src.preprocessing import (
    _find_common_prefix_boundary,
    clean_text,
    extract_hh_rlhf_triple,
)


class TestCleanText:
    def test_normalises_whitespace(self):
        assert clean_text("hello   world\n\n\tfoo") == "hello world foo"

    def test_strips_null_bytes(self):
        assert clean_text("foo\x00bar") == "foobar"

    def test_handles_empty(self):
        assert clean_text("") == ""

    def test_handles_none_safely(self):
        assert clean_text(None) == ""


class TestCommonPrefixBoundary:
    def test_identical_strings(self):
        s = "hello world"
        assert _find_common_prefix_boundary(s, s) == len(s)

    def test_no_overlap(self):
        assert _find_common_prefix_boundary("abc", "xyz") == 0

    def test_partial_overlap(self):
        assert _find_common_prefix_boundary("hello world", "hello there") == 6


class TestHHRLHFTripleExtraction:
    def test_basic_split(self):
        chosen = (
            "\n\nHuman: What is 2+2?\n\nAssistant: 4."
        )
        rejected = (
            "\n\nHuman: What is 2+2?\n\nAssistant: 5."
        )
        prompt, c, r = extract_hh_rlhf_triple(chosen, rejected)
        assert "What is 2+2?" in prompt
        assert "4" in c
        assert "5" in r
        # Chosen and rejected response strings differ at the divergence point
        assert c != r

    def test_multiturn(self):
        # Multi-turn: shared prefix includes one full turn before divergence
        prefix = "\n\nHuman: Hi.\n\nAssistant: Hello.\n\nHuman: What is 2+2?"
        chosen = prefix + "\n\nAssistant: 4."
        rejected = prefix + "\n\nAssistant: 5."
        prompt, c, r = extract_hh_rlhf_triple(chosen, rejected)
        assert "Hi." in prompt
        assert "Hello." in prompt
        assert "2+2" in prompt
        assert c.strip() != r.strip()
