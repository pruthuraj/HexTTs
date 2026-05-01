"""
Windows-native forced alignment using torchaudio WAV2VEC2_ASR_BASE_960H.

Replaces MFA for generating real duration targets on platforms where Kaldi
binaries are unavailable (Windows venv, no conda).

Output: per-file int32 .npy arrays in the same format as extract_mfa_durations.py.
Each array holds one integer per phoneme: the number of mel frames assigned to it.

Pipeline
--------
1. Load audio → resample to 16 kHz for wav2vec2
2. Run WAV2VEC2_ASR_BASE_960H → log-softmax emission
3. Encode transcript as character token indices (|  between words)
4. forced_align → per-frame token assignments
5. merge_tokens → token spans (char → time range)
6. Group char spans into per-word frame ranges
7. g2p_en per word → phoneme count per word
8. Distribute word frame counts uniformly across phonemes
9. Scale alignment frames → mel frames (proportional to total audio length)
10. Save as int32 .npy

Usage
-----
    python scripts/align_torchaudio.py \\
        --audio_dir    data/LJSpeech-1.1/wavs \\
        --metadata_csv data/LJSpeech-1.1/metadata.csv \\
        --prepared_dir data/ljspeech_prepared \\
        --output_dir   data/ljspeech_prepared/durations \\
        [--hop_length 256] [--orig_sr 22050] [--device cpu] [--limit 0]

After running, set in configs/base.yaml:
    duration_dir: ./data/ljspeech_prepared/durations
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

log = logging.getLogger(__name__)

# WAV2VEC2_ASR_BASE_960H constants
_ALIGN_SR = 16_000
_BLANK_IDX = 0  # '-' is index 0 in the bundle label list


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------

def build_label_index(bundle) -> Dict[str, int]:
    return {label: i for i, label in enumerate(bundle.get_labels())}


def encode_text(text: str, label_to_idx: Dict[str, int]) -> Optional[torch.Tensor]:
    """
    Encode text to token indices for forced_align.
    Inserts '|' word-boundary tokens between words.
    Unknown characters (punctuation, digits) are silently dropped.
    Returns None if no encodable characters remain.
    """
    tokens: List[int] = []
    words = text.upper().strip().split()
    for w_i, word in enumerate(words):
        chars = [label_to_idx[ch] for ch in word if ch in label_to_idx]
        tokens.extend(chars)
        if w_i < len(words) - 1 and '|' in label_to_idx:
            tokens.append(label_to_idx['|'])
    return torch.tensor(tokens, dtype=torch.long) if tokens else None


# ---------------------------------------------------------------------------
# Alignment → spans
# ---------------------------------------------------------------------------

def _collapse_alignment(
    token_seq: torch.Tensor,
    score_seq: torch.Tensor,
) -> List[dict]:
    """
    Fallback span extractor when torchaudio.functional.merge_tokens is absent.
    Groups consecutive identical non-blank frames into span dicts.
    End index is exclusive, matching torchaudio's TokenSpan convention.
    """
    spans: List[dict] = []
    prev_tok: Optional[int] = None
    span_start = 0
    span_scores: List[float] = []

    for t, (tok, sc) in enumerate(zip(token_seq.tolist(), score_seq.tolist())):
        if tok == _BLANK_IDX:
            if prev_tok is not None:
                spans.append({'token': prev_tok, 'start': span_start,
                               'end': t, 'score': float(np.mean(span_scores))})
                prev_tok = None
                span_scores = []
        elif tok != prev_tok:
            if prev_tok is not None:
                spans.append({'token': prev_tok, 'start': span_start,
                               'end': t, 'score': float(np.mean(span_scores))})
            prev_tok = tok
            span_start = t
            span_scores = [sc]
        else:
            span_scores.append(sc)

    if prev_tok is not None:
        spans.append({'token': prev_tok, 'start': span_start,
                       'end': len(token_seq), 'score': float(np.mean(span_scores))})
    return spans


def _span_attr(span, key: str):
    """Read attribute or dict key from either TokenSpan or dict."""
    return getattr(span, key, None) if hasattr(span, key) else span[key]


def get_word_frame_ranges(
    text: str,
    spans,
    label_to_idx: Dict[str, int],
) -> Optional[List[Tuple[int, int]]]:
    """
    Group character spans into per-word (start_frame, end_frame) pairs.
    end_frame is exclusive (matching torchaudio convention).
    Returns None if span count does not match expected character count.
    """
    PIPE_IDX = label_to_idx.get('|', -1)
    words = text.upper().strip().split()
    expected_spans = sum(
        sum(1 for ch in w if ch in label_to_idx) for w in words
    ) + max(0, len(words) - 1)  # +1 per '|' between words

    if len(spans) != expected_spans:
        return None

    word_ranges: List[Tuple[int, int]] = []
    span_idx = 0

    for w_i, word in enumerate(words):
        chars = [ch for ch in word if ch in label_to_idx]
        n = len(chars)
        if n == 0:
            return None  # word with no encodable chars (e.g. a number) → can't place it
        w_start = _span_attr(spans[span_idx], 'start')
        w_end = _span_attr(spans[span_idx + n - 1], 'end')
        word_ranges.append((w_start, w_end))
        span_idx += n
        if w_i < len(words) - 1:  # consume the '|' span
            span_idx += 1

    return word_ranges


# ---------------------------------------------------------------------------
# Phoneme distribution
# ---------------------------------------------------------------------------

def phonemes_per_word(text: str, g2p) -> Optional[List[int]]:
    """
    Return list of ARPAbet phoneme counts per word, using g2p_en.
    Uses the same text_to_phonemes helper as prepare_data.py.
    """
    try:
        from hextts.data.preprocessing import text_to_phonemes
        counts = []
        for word in text.strip().split():
            phones = text_to_phonemes(word, g2p)
            counts.append(max(1, len(phones.split()) if phones.strip() else 1))
        return counts
    except Exception:
        return None


def distribute_frames(
    word_align_frames: List[int],
    phones_per_word: List[int],
    total_mel_frames: int,
) -> np.ndarray:
    """
    Scale word-level alignment frame counts to mel space, then distribute
    uniformly within each word across its phonemes.
    Guarantees each phoneme gets >= 1 frame and the total equals total_mel_frames.
    """
    n_phones = sum(phones_per_word)
    total_align = max(1, sum(word_align_frames))
    scale = total_mel_frames / total_align

    # Scale to mel frames; floor to int but ensure >= n_phones_in_word
    mel_counts = [max(n, round(a * scale)) for a, n in zip(word_align_frames, phones_per_word)]

    # Adjust total to match exactly
    delta = total_mel_frames - sum(mel_counts)
    if delta > 0:
        mel_counts[-1] += delta
    elif delta < 0:
        # Trim excess from the longest words first
        for idx in sorted(range(len(mel_counts)), key=lambda i: -mel_counts[i]):
            cut = min(-delta, mel_counts[idx] - phones_per_word[idx])
            mel_counts[idx] -= cut
            delta += cut
            if delta == 0:
                break

    # Distribute each word's frames uniformly across its phonemes
    result: List[int] = []
    for frames, n_ph in zip(mel_counts, phones_per_word):
        base = frames // n_ph
        rem = frames % n_ph
        result.extend(base + (1 if i < rem else 0) for i in range(n_ph))

    return np.array(result, dtype=np.int32)


def uniform_fallback(total_mel_frames: int, n_phonemes: int) -> np.ndarray:
    """Distribute total_mel_frames uniformly across n_phonemes."""
    base = max(1, total_mel_frames // n_phonemes)
    arr = np.full(n_phonemes, base, dtype=np.int32)
    rem = max(0, total_mel_frames - base * n_phonemes)
    arr[:rem] += 1
    return arr


# ---------------------------------------------------------------------------
# Per-file alignment
# ---------------------------------------------------------------------------

def align_file(
    wav_path: str,
    text: str,
    n_phonemes: int,
    model,
    label_to_idx: Dict[str, int],
    g2p,
    hop_length: int,
    orig_sr: int,
    device: torch.device,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Align one audio file. Returns (duration_array, used_real_alignment).
    duration_array is None on hard failure (file skipped entirely).
    used_real_alignment=False means uniform fallback was used (file saved but low quality).
    """
    try:
        waveform, sr = torchaudio.load(wav_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # force mono
    except Exception as e:
        log.warning("Load failed %s: %s", wav_path, e)
        return None, False

    # Total mel frames from original audio (ground truth for scaling)
    n_samples = int(waveform.shape[-1] * orig_sr / sr) if sr != orig_sr else waveform.shape[-1]
    total_mel_frames = max(n_phonemes, (n_samples + hop_length - 1) // hop_length)

    # Resample for wav2vec2
    if sr != _ALIGN_SR:
        waveform = torchaudio.functional.resample(waveform, sr, _ALIGN_SR)

    targets_1d = encode_text(text, label_to_idx)
    if targets_1d is None:
        return uniform_fallback(total_mel_frames, n_phonemes), False

    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        log_probs = torch.log_softmax(emission, dim=-1).cpu()  # (1, T, C)

    T = log_probs.size(1)
    S = targets_1d.size(0)
    if S >= T:
        log.debug("Transcript too long vs audio T=%d S=%d for %s", T, S, wav_path)
        return uniform_fallback(total_mel_frames, n_phonemes), False

    try:
        token_seq, score_seq = torchaudio.functional.forced_align(
            log_probs,
            targets_1d.unsqueeze(0),
            torch.tensor([T], dtype=torch.long),
            torch.tensor([S], dtype=torch.long),
            blank=_BLANK_IDX,
        )
    except Exception as e:
        log.warning("forced_align error for %s: %s", wav_path, e)
        return uniform_fallback(total_mel_frames, n_phonemes), False

    # Collapse per-frame assignments into token spans
    try:
        spans = torchaudio.functional.merge_tokens(token_seq[0], score_seq[0])
    except AttributeError:
        spans = _collapse_alignment(token_seq[0], score_seq[0])

    word_ranges = get_word_frame_ranges(text, spans, label_to_idx)
    if word_ranges is None:
        return uniform_fallback(total_mel_frames, n_phonemes), False

    ppw = phonemes_per_word(text, g2p) if g2p is not None else None
    if ppw is None or sum(ppw) != n_phonemes or len(ppw) != len(word_ranges):
        return uniform_fallback(total_mel_frames, n_phonemes), False

    word_align_frames = [max(1, end - start) for start, end in word_ranges]
    dur = distribute_frames(word_align_frames, ppw, total_mel_frames)

    if len(dur) != n_phonemes:
        return uniform_fallback(total_mel_frames, n_phonemes), False

    return dur, True


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_prepared_metadata(prepared_dir: str) -> Dict[str, int]:
    """Load train.txt + val.txt → {filename: n_phonemes}."""
    mapping: Dict[str, int] = {}
    for split in ("train.txt", "val.txt"):
        path = os.path.join(prepared_dir, split)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('|', 1)
                if len(parts) == 2:
                    mapping[parts[0]] = len(parts[1].strip().split())
    return mapping


def load_ljspeech_csv(metadata_csv: str) -> Dict[str, str]:
    """
    Load LJSpeech metadata.csv → {filename: text}.
    Uses column 1 (same column as preprocessing.py) for text.
    """
    mapping: Dict[str, str] = {}
    with open(metadata_csv, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter='|'):
            if len(row) >= 2:
                mapping[row[0].strip()] = row[1].strip()
    return mapping


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="WAV2VEC2-based forced alignment → per-phoneme duration .npy files"
    )
    parser.add_argument("--audio_dir",    default="data/LJSpeech-1.1/wavs")
    parser.add_argument("--metadata_csv", default="data/LJSpeech-1.1/metadata.csv")
    parser.add_argument("--prepared_dir", default="data/ljspeech_prepared")
    parser.add_argument("--output_dir",   default="data/ljspeech_prepared/durations")
    parser.add_argument("--hop_length",   type=int, default=256)
    parser.add_argument("--orig_sr",      type=int, default=22050)
    parser.add_argument("--device",       default="cpu",
                        help="'cuda' is faster for large datasets")
    parser.add_argument("--limit",        type=int, default=0,
                        help="Process only the first N files (0 = all)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    log.info("Loading WAV2VEC2_ASR_BASE_960H ...")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device).eval()
    label_to_idx = build_label_index(bundle)
    log.info("Labels: %s...", bundle.get_labels()[:6])

    log.info("Loading g2p_en ...")
    try:
        from g2p_en import G2p
        g2p = G2p()
        log.info("g2p_en ready.")
    except ImportError:
        log.warning("g2p_en not installed — per-word distribution unavailable, "
                    "all files will use uniform fallback. Run: pip install g2p_en")
        g2p = None

    log.info("Reading metadata ...")
    prepared = load_prepared_metadata(args.prepared_dir)
    texts = load_ljspeech_csv(args.metadata_csv)
    log.info("Prepared: %d files  |  LJSpeech CSV: %d files", len(prepared), len(texts))

    filenames = sorted(set(prepared) & set(texts))
    if args.limit > 0:
        filenames = filenames[:args.limit]
    log.info("Processing %d files ...", len(filenames))

    n_real = n_uniform = n_skip = 0

    for i, fname in enumerate(filenames):
        out_path = out_dir / f"{fname}.npy"
        if out_path.exists():
            n_real += 1
            continue

        wav_path = os.path.join(args.audio_dir, f"{fname}.wav")
        if not os.path.exists(wav_path):
            n_skip += 1
            continue

        dur, is_real = align_file(
            wav_path=wav_path,
            text=texts[fname],
            n_phonemes=prepared[fname],
            model=model,
            label_to_idx=label_to_idx,
            g2p=g2p,
            hop_length=args.hop_length,
            orig_sr=args.orig_sr,
            device=device,
        )

        if dur is None:
            n_skip += 1
            log.warning("[%d/%d] SKIP %s", i + 1, len(filenames), fname)
            continue

        if len(dur) != prepared[fname]:
            n_skip += 1
            log.warning("[%d/%d] phoneme mismatch %s: got %d expected %d",
                        i + 1, len(filenames), fname, len(dur), prepared[fname])
            continue

        np.save(str(out_path), dur)
        if is_real:
            n_real += 1
        else:
            n_uniform += 1

        if (i + 1) % 500 == 0 or i == len(filenames) - 1:
            log.info("[%d/%d]  real=%d  uniform-fallback=%d  skipped=%d",
                     i + 1, len(filenames), n_real, n_uniform, n_skip)

    total_saved = n_real + n_uniform
    log.info("")
    log.info("Done.  Saved: %d  (real=%d  fallback=%d)  Skipped: %d",
             total_saved, n_real, n_uniform, n_skip)
    log.info("Real alignment coverage: %.1f%%",
             100 * n_real / max(1, len(filenames)))
    log.info("")
    log.info("Next step — set in configs/base.yaml:")
    log.info("    duration_dir: %s", args.output_dir.replace("\\", "/"))


if __name__ == "__main__":
    main()
