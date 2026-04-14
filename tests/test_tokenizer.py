"""Tests for phoneme-to-ID normalization and determinism."""

from hextts.inference.text_processing import phonemes_to_ids


def test_phonemes_to_ids_handles_unknown_and_empty():
    """Unknown symbols should not crash conversion and should still yield at least one ID."""
    ids = phonemes_to_ids("AH0 ZZ_UNKNOWN")
    assert isinstance(ids, list)
    assert len(ids) >= 1


def test_phonemes_to_ids_is_deterministic():
    """Tokenization must be deterministic for reproducible training/inference behavior."""
    s = "AH0 B K"
    assert phonemes_to_ids(s) == phonemes_to_ids(s)
