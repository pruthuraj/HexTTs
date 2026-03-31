"""
Convert LJSpeech metadata to VITS format with phonemes
Generates train/val splits and phoneme sequences
"""

import os
import csv
import json
import random
from pathlib import Path
from tqdm import tqdm

try:
    from g2p_en import G2p
except ImportError:
    print("Error: g2p_en not installed. Run: pip install g2p_en")
    exit(1)


def text_to_phonemes(text, g2p):
    """
    Convert text to normalized ARPAbet phoneme sequence using g2p_en.

    Example:
        "Hello" -> "HH AH L OW"
    """
    try:
        phoneme_list = g2p(text)

        normalized = []
        for token in phoneme_list:
            token = token.strip()
            if not token or token == " ":
                continue

            # Remove ARPAbet stress markers: AH0 -> AH, EH1 -> EH
            token = token.upper().rstrip("012")

            # Keep only alphabetic ARPAbet-like tokens
            if token.isalpha():
                normalized.append(token)

        return " ".join(normalized)
    except Exception as e:
        print(f"Warning: Error converting '{text}': {e}")
        return ""


def process_ljspeech_metadata(dataset_path, output_path, train_split=0.95, seed=42):
    """
    Process LJSpeech metadata and convert to VITS phoneme format.
    Output format:
        filename|PHONEME PHONEME PHONEME
    """

    print("=" * 60)
    print("LJSpeech to VITS Format Conversion")
    print("=" * 60)

    random.seed(seed)

    metadata_file = os.path.join(dataset_path, "metadata.csv")
    wavs_dir = os.path.join(dataset_path, "wavs")

    if not os.path.exists(metadata_file):
        print(f"Error: {metadata_file} not found!")
        return False

    if not os.path.exists(wavs_dir):
        print(f"Error: {wavs_dir} not found!")
        return False

    os.makedirs(output_path, exist_ok=True)

    print(f"Reading metadata from: {metadata_file}")
    print(f"Output directory: {output_path}\n")

    print("Initializing phoneme converter (g2p_en)...")
    g2p = G2p()
    print("Phoneme converter ready\n")

    metadata = []
    with open(metadata_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        metadata = list(reader)

    print(f"Processing {len(metadata)} utterances...\n")

    processed_data = []
    skipped_missing_audio = 0
    skipped_empty_phonemes = 0

    for i, row in enumerate(tqdm(metadata, desc="Converting to phonemes")):
        if len(row) < 3:
            print(f"Warning: Row {i} has only {len(row)} columns, skipping")
            continue

        filename = row[0]
        normalized_text = row[1]
        raw_text = row[2]

        wav_path = os.path.join(wavs_dir, f"{filename}.wav")
        if not os.path.exists(wav_path):
            skipped_missing_audio += 1
            continue

        phonemes = text_to_phonemes(normalized_text, g2p)
        if not phonemes.strip():
            skipped_empty_phonemes += 1
            continue

        processed_data.append({
            "filename": filename,
            "text": normalized_text,
            "raw_text": raw_text,
            "phonemes": phonemes,
        })

    print(f"\nSuccessfully processed {len(processed_data)} utterances")
    print(f"Skipped missing audio: {skipped_missing_audio}")
    print(f"Skipped empty phonemes: {skipped_empty_phonemes}")

    if not processed_data:
        print("Error: No valid processed samples found.")
        return False

    random.shuffle(processed_data)

    num_train = int(len(processed_data) * train_split)
    train_data = processed_data[:num_train]
    val_data = processed_data[num_train:]

    print(f"\nDataset split:")
    print(f"  Training: {len(train_data)} ({100 * len(train_data) / len(processed_data):.1f}%)")
    print(f"  Validation: {len(val_data)} ({100 * len(val_data) / len(processed_data):.1f}%)")

    train_file = os.path.join(output_path, "train.txt")
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(f"{item['filename']}|{item['phonemes']}\n")

    val_file = os.path.join(output_path, "val.txt")
    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(f"{item['filename']}|{item['phonemes']}\n")

    metadata_json = os.path.join(output_path, "metadata.json")
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(processed_data),
            "train": len(train_data),
            "val": len(val_data),
            "skipped_missing_audio": skipped_missing_audio,
            "skipped_empty_phonemes": skipped_empty_phonemes,
            "sample": processed_data[:5]
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSaved training metadata to: {train_file}")
    print(f"Saved validation metadata to: {val_file}")
    print(f"Saved detailed metadata to: {metadata_json}")

    print("\n" + "=" * 60)
    print("SAMPLE CONVERSIONS")
    print("=" * 60)

    for i in range(min(5, len(processed_data))):
        item = processed_data[i]
        print(f"\nFile: {item['filename']}")
        print(f"Text:     {item['text']}")
        print(f"Phonemes: {item['phonemes']}")

    print("\n" + "=" * 60)
    print("Conversion complete")
    print("=" * 60)

    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "./data/LJSpeech-1.1"

    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = "./data/ljspeech_prepared"

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        print("Usage: python prepare_data.py <dataset_path> <output_path>")
        print("Example: python prepare_data.py ./data/LJSpeech-1.1 ./data/ljspeech_prepared")
        sys.exit(1)

    success = process_ljspeech_metadata(dataset_path, output_path)
    sys.exit(0 if success else 1)
