"""Annotation whitelist and AAMI mapping tests."""

from src.data.segment import (
    BEAT_SYMBOL_TO_AAMI,
    NON_BEAT_SYMBOLS,
    map_to_aami_class,
)


def test_known_beats_map():
    assert map_to_aami_class("N") == "N"
    assert map_to_aami_class("V") == "V"
    assert map_to_aami_class("A") == "S"
    assert map_to_aami_class("F") == "F"
    assert map_to_aami_class("/") == "Q"


def test_non_beat_skipped():
    for sym in ["+", "~", "p", "t", "(", ")"]:
        assert map_to_aami_class(sym) is None
        assert sym in NON_BEAT_SYMBOLS


def test_unknown_not_forced_to_q():
    # Critical: unknown must not silently become Q
    assert map_to_aami_class("???") is None
    assert map_to_aami_class("Z") is None


def test_whitelist_covers_supported_classes():
    mapped = set(BEAT_SYMBOL_TO_AAMI.values())
    assert mapped == {"N", "V", "S", "F", "Q"}
