"""Thin wrapper for package-based dataset preprocessing."""

from hextts.data.preprocessing import cli_main, process_ljspeech_metadata, text_to_phonemes


if __name__ == "__main__":
    import sys

    sys.exit(cli_main())
