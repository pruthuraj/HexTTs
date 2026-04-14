"""Text processing helpers for inference."""

from __future__ import annotations

from typing import List

from vits_data import PHONEME_TO_ID


def phonemes_to_ids(phoneme_str: str) -> List[int]:
    ids = []
    for token in phoneme_str.split():
        p = token.strip().upper().rstrip("012")
        if not p:
            continue
        if p in PHONEME_TO_ID:
            ids.append(PHONEME_TO_ID[p])

    if not ids:
        ids = [PHONEME_TO_ID["PAD"]]

    return ids
