"""Thin wrapper for package-based cache feature precompute."""

from hextts.data.cache_builder import (
    audio_to_mel,
    cli_main,
    phonemes_to_ids,
    process_split,
)


if __name__ == "__main__":
    raise SystemExit(cli_main())
