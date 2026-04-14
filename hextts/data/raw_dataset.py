"""Raw dataset compatibility layer."""

from vits_data import (  # noqa: F401
    TTSDataset,
    PHONEME_TO_ID,
    ID_TO_PHONEME,
    VOCAB_SIZE,
    record_warning,
    get_warning_summary,
    reset_warning_summary,
)
