from hextts.inference.text_processing import phonemes_to_ids


def test_phonemes_to_ids_handles_unknown_and_empty():
    ids = phonemes_to_ids("AH0 ZZ_UNKNOWN")
    assert isinstance(ids, list)
    assert len(ids) >= 1


def test_phonemes_to_ids_is_deterministic():
    s = "AH0 B K"
    assert phonemes_to_ids(s) == phonemes_to_ids(s)
