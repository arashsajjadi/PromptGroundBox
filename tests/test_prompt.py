from __future__ import annotations

import pytest

from promptgroundboxbench.utils.prompt import normalize_label, parse_prompt


def test_parse_prompt_periods() -> None:
    labels = parse_prompt("a person. a chair. a laptop.")
    assert labels == ["a person", "a chair", "a laptop"]


def test_parse_prompt_commas_newlines() -> None:
    labels = parse_prompt("cat, dog\nbird; horse")
    assert labels == ["cat", "dog", "bird", "horse"]


def test_parse_prompt_dedup_case_insensitive() -> None:
    labels = parse_prompt("Cat. cat. CAT.")
    assert labels == ["Cat"]


def test_parse_prompt_empty() -> None:
    assert parse_prompt("...   ; \n") == []


def test_normalize_label_articles_and_punct() -> None:
    assert normalize_label("a Person.") == "person"
    assert normalize_label("the traffic light") == "traffic light"
