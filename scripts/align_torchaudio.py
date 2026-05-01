"""
Windows-native forced alignment using torchaudio WAV2VEC2_ASR_BASE_960H.

Replaces MFA for generating real duration targets on platforms where Kaldi
binaries are unavailable (Windows venv, no conda).

Output: per-file int32 .npy arrays in the same format as extract_mfa_durations.py.
Each array holds one integer per phoneme: the number of mel frames assigned to it.

Pipeline
--------
1. DataLoader (num_workers) loads + resamples audio in parallel
2. Pad batch → batched wav2vec2 inference (GPU-friendly)
3. Per-file: forced_align → merge_tokens → word frame ranges
4. Per-word: g2p (word-level cache) → distribute frames to phonemes
5. Save as int32 .npy

Usage
-----
    python scripts/align_torchaudio.py \\
        --audio_dir    data/LJSpeech-1.1/wavs \\
        --metadata_csv data/LJSpeech-1.1/metadata.csv \\
        --prepared_dir data/ljspeech_prepared \\
        --output_dir   data/ljspeech_prepared/durations \\
        [--device cuda] [--batch_size 8] [--num_workers 2] [--limit 0]

After running, set in configs/base.yaml:
    duration_dir: ./data/ljspeech_prepared/durations
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path so hextts can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

log = logging.getLogger(__name__)

_ALIGN_SR = 16_000
_BLANK_IDX = 0  # '-' is index 0 in WAV2VEC2_ASR_BASE_960H labels

# Word-level g2p cache: word → phoneme count.  Shared across the whole run.
_WORD_CACHE: Dict[str, int] = {}


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------

def build_label_index(bundle) -> Dict[str, int]:
    return {label: i for i, label in enumerate(bundle.get_labels())}


def _is_content_char(ch: str, label_to_idx: Dict[str, int]) -> bool:
    """True if ch maps to a non-blank label (index > 0)."""
    return label_to_idx.get(ch, 0) > 0


def _encoded_words(text: str, label_to_idx: Dict[str, int]) -> List[List[int]]:
    """Return per-word encoded char IDs, dropping words with no encodable chars."""
    out: List[List[int]] = []
    for raw_word in text.upper().strip().split():
        chars = [label_to_idx[ch] for ch in raw_word if _is_content_char(ch, label_to_idx)]
        if chars:
            out.append(chars)
    return out


def _g2p_words(text: str) -> List[str]:
    """Normalize raw words for g2p to reduce punctuation-related mismatches."""
    words: List[str] = []
    for raw_word in text.strip().split():
        clean = re.sub(r"[^A-Za-z']+", "", raw_word)
        if clean:
            words.append(clean)
    return words


def encode_text(text: str, label_to_idx: Dict[str, int]) -> Optional[torch.Tensor]:
    """
    Encode text to token indices for forced_align.
    Inserts '|' word-boundary tokens between words.
    Unknown characters and '-' (blank, index 0) are silently dropped.
    Returns None if no encodable characters remain.
    """
    tokens: List[int] = []
    words = _encoded_words(text, label_to_idx)
    for w_i, chars in enumerate(words):
        tokens.extend(chars)
        if w_i < len(words) - 1 and '|' in label_to_idx:
            tokens.append(label_to_idx['|'])
    return torch.tensor(tokens, dtype=torch.long) if tokens else None


# ---------------------------------------------------------------------------
# Alignment → spans
# ---------------------------------------------------------------------------

def _collapse_alignment(token_seq: torch.Tensor, score_seq: torch.Tensor) -> List[dict]:
    """Fallback: collapse per-frame assignments into span dicts (end exclusive)."""
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
    return getattr(span, key, None) if hasattr(span, key) else span[key]


def _span_int_attr(span, key: str) -> Optional[int]:
    value = _span_attr(span, key)
    if value is None:
        return None
    return int(value)


def _group_word_ranges_from_content_spans(
    content_spans,
    chars_per_word: List[int],
) -> Optional[List[Tuple[int, int]]]:
    """Group non-pipe spans into word ranges, with a proportional fallback if sparse."""
    if not content_spans or not chars_per_word:
        return None

    total_chars = sum(chars_per_word)
    n_spans = len(content_spans)

    if n_spans >= total_chars:
        ranges: List[Tuple[int, int]] = []
        idx = 0
        for n_chars in chars_per_word:
            start = _span_int_attr(content_spans[idx], 'start')
            end = _span_int_attr(content_spans[idx + n_chars - 1], 'end')
            if start is None or end is None:
                return None
            ranges.append((start, end))
            idx += n_chars
        return ranges

    # Sparse spans (e.g., merged repeats): allocate available spans proportionally.
    alloc = [max(1, round(n_spans * c / max(1, total_chars))) for c in chars_per_word]
    delta = n_spans - sum(alloc)
    if delta > 0:
        alloc[-1] += delta
    elif delta < 0:
        for i in sorted(range(len(alloc)), key=lambda j: -alloc[j]):
            cut = min(-delta, alloc[i] - 1)
            alloc[i] -= cut
            delta += cut
            if delta == 0:
                break

    if sum(alloc) != n_spans:
        return None

    ranges = []
    idx = 0
    for n_take in alloc:
        start = _span_int_attr(content_spans[idx], 'start')
        end = _span_int_attr(content_spans[idx + n_take - 1], 'end')
        if start is None or end is None:
            return None
        ranges.append((start, end))
        idx += n_take
    return ranges


def get_word_frame_ranges(
    text: str,
    spans,
    label_to_idx: Dict[str, int],
) -> Optional[List[Tuple[int, int]]]:
    """Group character spans into per-word frame ranges (end exclusive)."""
    encoded_words = _encoded_words(text, label_to_idx)
    chars_per_word = [len(chars) for chars in encoded_words]
    if not chars_per_word:
        return None

    pipe_idx = label_to_idx.get('|', None)
    content_spans = []
    for sp in spans:
        tok = _span_attr(sp, 'token')
        if tok == _BLANK_IDX:
            continue
        if pipe_idx is not None and tok == pipe_idx:
            continue
        content_spans.append(sp)

    return _group_word_ranges_from_content_spans(content_spans, chars_per_word)


# ---------------------------------------------------------------------------
# Phoneme distribution (with word-level cache)
# ---------------------------------------------------------------------------

def _phones_for_word(word: str, g2p, text_to_phonemes_fn) -> int:
    if word not in _WORD_CACHE:
        phones = text_to_phonemes_fn(word, g2p)
        _WORD_CACHE[word] = max(1, len(phones.split()) if phones.strip() else 1)
    return _WORD_CACHE[word]


def phonemes_per_word(text: str, g2p) -> Optional[List[int]]:
    """Per-word phoneme counts using g2p_en with word-level caching."""
    try:
        from hextts.data.preprocessing import text_to_phonemes
        words = _g2p_words(text)
        if not words:
            return None
        return [_phones_for_word(w, g2p, text_to_phonemes) for w in words]
    except Exception as e:
        log.warning("phonemes_per_word exception: %s", e)
        return None


def _fit_ppw_to_total(ppw: List[int], total_phonemes: int) -> Optional[List[int]]:
    """Adjust per-word phoneme counts so sum(ppw)==total_phonemes, keeping >=1 each."""
    if not ppw:
        return None
    n_words = len(ppw)
    if total_phonemes < n_words:
        return None

    fitted = [max(1, int(x)) for x in ppw]
    delta = total_phonemes - sum(fitted)
    if delta > 0:
        for i in sorted(range(n_words), key=lambda j: -fitted[j]):
            add = max(1, delta // max(1, n_words))
            fitted[i] += add
            delta -= add
            if delta <= 0:
                break
        if delta > 0:
            fitted[-1] += delta
    elif delta < 0:
        need = -delta
        for i in sorted(range(n_words), key=lambda j: -fitted[j]):
            cut = min(need, fitted[i] - 1)
            fitted[i] -= cut
            need -= cut
            if need == 0:
                break
        if need > 0:
            return None

    return fitted if sum(fitted) == total_phonemes else None


def distribute_frames(
    word_align_frames: List[int],
    phones_per_word: List[int],
    total_mel_frames: int,
) -> np.ndarray:
    """Scale word frame counts to mel space and distribute uniformly across phonemes."""
    total_align = max(1, sum(word_align_frames))
    scale = total_mel_frames / total_align

    mel_counts = [max(n, round(a * scale)) for a, n in zip(word_align_frames, phones_per_word)]

    delta = total_mel_frames - sum(mel_counts)
    if delta > 0:
        mel_counts[-1] += delta
    elif delta < 0:
        for idx in sorted(range(len(mel_counts)), key=lambda i: -mel_counts[i]):
            cut = min(-delta, mel_counts[idx] - phones_per_word[idx])
            mel_counts[idx] -= cut
            delta += cut
            if delta == 0:
                break

    result: List[int] = []
    for frames, n_ph in zip(mel_counts, phones_per_word):
        base = frames // n_ph
        rem = frames % n_ph
        result.extend(base + (1 if i < rem else 0) for i in range(n_ph))
    return np.array(result, dtype=np.int32)


def uniform_fallback(total_mel_frames: int, n_phonemes: int) -> np.ndarray:
    base = max(1, total_mel_frames // n_phonemes)
    arr = np.full(n_phonemes, base, dtype=np.int32)
    rem = max(0, total_mel_frames - base * n_phonemes)
    arr[:rem] += 1
    return arr


# ---------------------------------------------------------------------------
# Core alignment (given pre-computed log_probs for one file)
# ---------------------------------------------------------------------------

def _align_with_log_probs(
    log_probs: torch.Tensor,   # (1, T, C) — already sliced to valid frames
    text: str,
    n_phonemes: int,
    total_mel_frames: int,
    label_to_idx: Dict[str, int],
    g2p,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Run forced alignment and phoneme distribution for one file.
    Returns (duration_array, used_real_alignment).
    duration_array is None only on hard failure (file should be skipped entirely).
    """
    targets_1d = encode_text(text, label_to_idx)
    if targets_1d is None:
        return uniform_fallback(total_mel_frames, n_phonemes), False

    T = log_probs.size(1)
    S = targets_1d.size(0)
    if S >= T:
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
        log.warning("forced_align error: %s", e)
        return uniform_fallback(total_mel_frames, n_phonemes), False

    try:
        spans = torchaudio.functional.merge_tokens(token_seq[0], score_seq[0])
    except AttributeError:
        spans = _collapse_alignment(token_seq[0], score_seq[0])

    word_ranges = get_word_frame_ranges(text, spans, label_to_idx)
    if word_ranges is None:
        return uniform_fallback(total_mel_frames, n_phonemes), False

    ppw = phonemes_per_word(text, g2p) if g2p is not None else None
    if ppw is None:
        return uniform_fallback(total_mel_frames, n_phonemes), False

    # If word count mismatch, try to redistribute phonemes proportionally to alignment
    if len(ppw) != len(word_ranges):
        if len(word_ranges) == 0:
            return uniform_fallback(total_mel_frames, n_phonemes), False
        word_align_frames = [max(1, end - start) for start, end in word_ranges]
        total_wf = max(1, sum(word_align_frames))
        ppw = [max(1, round(n_phonemes * wf / total_wf)) for wf in word_align_frames]

    # Adjust ppw to match target total
    current_total = sum(ppw)
    if current_total != n_phonemes:
        delta = n_phonemes - current_total
        if delta > 0:
            ppw[-1] += delta
        else:
            ppw[-1] = max(1, ppw[-1] + delta)

    word_align_frames = [max(1, end - start) for start, end in word_ranges]
    dur = distribute_frames(word_align_frames, ppw, total_mel_frames)
    if dur is None or len(dur) != n_phonemes:
        return uniform_fallback(total_mel_frames, n_phonemes), False

    return dur, True


# ---------------------------------------------------------------------------
# Dataset (parallel audio loading via DataLoader workers)
# ---------------------------------------------------------------------------

class _AudioDataset(Dataset):
    """Loads and resamples one audio file per item. Safe to use with num_workers > 0."""

    def __init__(self, items: List[Tuple[str, str, str, int]], orig_sr: int):
        # items: [(fname, wav_path, text, n_phonemes), ...]
        self.items = items
        self.orig_sr = orig_sr

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        fname, wav_path, text, n_phonemes = self.items[idx]
        try:
            waveform, sr = torchaudio.load(wav_path)
            waveform = waveform.mean(0)  # (L,) mono
            # Compute mel frame count from original sample rate
            n_orig = waveform.shape[0] if sr == self.orig_sr else int(waveform.shape[0] * self.orig_sr / sr)
            if sr != _ALIGN_SR:
                waveform = torchaudio.functional.resample(waveform, sr, _ALIGN_SR)
            return fname, waveform, text, n_phonemes, n_orig, True
        except Exception:
            return fname, torch.zeros(1), text, n_phonemes, 0, False


def _collate_audio(batch):
    """Pad waveforms in a batch to the same length for batched wav2vec2 inference."""
    fnames, waves, texts, n_phones_list, n_orig_list, ok_flags = zip(*batch)

    failed = [f for f, ok in zip(fnames, ok_flags) if not ok]
    valid = [(f, w, t, n, o) for f, w, t, n, o, ok in
             zip(fnames, waves, texts, n_phones_list, n_orig_list, ok_flags) if ok]

    if not valid:
        return [], None, None, [], [], [], failed

    vf, vw, vt, vn, vo = zip(*valid)
    lengths = torch.tensor([w.shape[0] for w in vw], dtype=torch.long)
    padded = torch.zeros(len(vw), int(lengths.max()))
    for i, w in enumerate(vw):
        padded[i, :w.shape[0]] = w

    return list(vf), padded, lengths, list(vt), list(vn), list(vo), failed


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
    """Load LJSpeech metadata.csv → {filename: text} using column 1 (matches preprocessing.py)."""
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
                        help="Use 'cuda' for ~5x speedup on GPU")
    parser.add_argument("--batch_size",   type=int, default=4,
                        help="Files per wav2vec2 forward pass. 8-16 recommended on GPU.")
    parser.add_argument("--num_workers",  type=int, default=2,
                        help="DataLoader workers for parallel audio loading (0 = main thread)")
    parser.add_argument("--prefetch_factor", type=int, default=4,
                        help="DataLoader prefetch factor when num_workers > 0")
    parser.add_argument("--disable_amp", action="store_true",
                        help="Disable AMP fast path on CUDA")
    parser.add_argument("--limit",        type=int, default=0,
                        help="Process only the first N files (0 = all)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_amp = (device.type == "cuda") and (not args.disable_amp)

    log.info("Loading WAV2VEC2_ASR_BASE_960H ...")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device).eval()
    label_to_idx = build_label_index(bundle)
    if device.type == "cuda":
        # Safe CUDA speedups for inference-heavy workloads.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log.info("Device: %s  batch_size: %d  num_workers: %d",
             args.device, args.batch_size, args.num_workers)
    log.info("AMP: %s", "on" if use_amp else "off")

    log.info("Loading g2p_en ...")
    try:
        from g2p_en import G2p
        g2p = G2p()
        log.info("g2p_en ready.")
    except ImportError:
        log.warning("g2p_en not installed — all files will use uniform fallback. "
                    "Run: pip install g2p_en")
        g2p = None

    log.info("Reading metadata ...")
    prepared = load_prepared_metadata(args.prepared_dir)
    texts = load_ljspeech_csv(args.metadata_csv)
    log.info("Prepared: %d  |  LJSpeech CSV: %d", len(prepared), len(texts))

    filenames = sorted(set(prepared) & set(texts))
    if args.limit > 0:
        filenames = filenames[:args.limit]
    log.info("Total files to consider: %d", len(filenames))

    # Split into already-done and pending
    work_items: List[Tuple[str, str, str, int]] = []
    n_real = n_uniform = n_skip = 0
    for fname in filenames:
        if (out_dir / f"{fname}.npy").exists():
            n_real += 1
            continue
        wav_path = os.path.join(args.audio_dir, f"{fname}.wav")
        if not os.path.exists(wav_path):
            n_skip += 1
        else:
            work_items.append((fname, wav_path, texts[fname], prepared[fname]))

    log.info("Already done: %d  |  To process: %d  |  Missing audio: %d",
             n_real, len(work_items), n_skip)

    dataset = _AudioDataset(work_items, orig_sr=args.orig_sr)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=_collate_audio,
        pin_memory=(args.device != "cpu"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )

    pbar = tqdm(total=len(filenames), unit="file", desc="Aligning", dynamic_ncols=True)
    pbar.update(n_real + n_skip)  # fast-forward past already-handled files
    processed_since_postfix = 0

    for batch in loader:
        vf, padded, lengths, vt, vn, vo, failed = batch

        for fname in failed:
            n_skip += 1
            tqdm.write(f"LOAD ERROR {fname}")
            pbar.update(1)

        if not vf:
            continue

        # Batched wav2vec2 inference — one GPU forward pass for the whole batch
        with torch.inference_mode():
            x = padded.to(device, non_blocking=(device.type == "cuda"))
            x_lengths = lengths.to(device, non_blocking=(device.type == "cuda"))
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp else contextlib.nullcontext()
            )
            with amp_ctx:
                emissions, out_lengths = model(x, lengths=x_lengths)
            log_probs_batch = torch.log_softmax(emissions.float(), dim=-1).cpu()
            out_lengths = out_lengths.cpu()

        # Per-file alignment (CPU, can't be batched due to different targets)
        for i, (fname, text, n_phones, n_orig) in enumerate(zip(vf, vt, vn, vo)):
            total_mel = max(n_phones, (n_orig + args.hop_length - 1) // args.hop_length)
            valid_T = int(out_lengths[i].item())
            file_lp = log_probs_batch[i:i+1, :valid_T, :]

            dur, is_real = _align_with_log_probs(
                file_lp, text, n_phones, total_mel, label_to_idx, g2p
            )

            if dur is None:
                n_skip += 1
                tqdm.write(f"SKIP {fname}")
            elif len(dur) != n_phones:
                n_skip += 1
                tqdm.write(f"MISMATCH {fname}: got {len(dur)} expected {n_phones}")
            else:
                np.save(str(out_dir / f"{fname}.npy"), dur)
                if is_real:
                    n_real += 1
                else:
                    n_uniform += 1

            pbar.update(1)
            processed_since_postfix += 1
            if processed_since_postfix >= 32:
                pbar.set_postfix(real=n_real, fallback=n_uniform, skip=n_skip)
                processed_since_postfix = 0

    pbar.close()
    total_saved = n_real + n_uniform
    log.info("Done.  Saved: %d  (real=%d  fallback=%d)  Skipped: %d",
             total_saved, n_real, n_uniform, n_skip)
    log.info("Real alignment coverage: %.1f%%", 100 * n_real / max(1, len(filenames)))
    log.info("")
    log.info("Next: set in configs/base.yaml:")
    log.info("    duration_dir: %s", args.output_dir.replace("\\", "/"))


if __name__ == "__main__":
    main()
