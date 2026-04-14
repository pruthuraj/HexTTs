"""Text processing helpers for inference."""

from __future__ import annotations

from typing import List

from hextts.data.raw_dataset import PHONEME_TO_ID


def phonemes_to_ids(phoneme_str: str) -> List[int]:
    """Convert a whitespace-separated phoneme string into model token IDs.

    The function normalizes stress markers (e.g. AH0/AH1/AH2 -> AH), drops empty
    tokens, and returns a PAD token when no valid phonemes remain.
    """
    ids = []
    for token in phoneme_str.split():
        # Normalize token format to match dataset vocabulary conventions.
        p = token.strip().upper().rstrip("012")
        if not p:
            continue
        if p in PHONEME_TO_ID:
            ids.append(PHONEME_TO_ID[p])

    if not ids:
        # Keep downstream model calls valid even for empty/unrecognized input.
        ids = [PHONEME_TO_ID["PAD"]]

    return ids
