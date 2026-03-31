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
    Convert text to phoneme sequence using g2p_en
    
    Example:
        "Hello" -> "HH AE L OW"
    """
    try:
        phoneme_list = g2p(text)
        # Remove stress markers (1, 2) and join
        phonemes = [p.rstrip('012') for p in phoneme_list]
        # Filter out special characters
        phonemes = [p for p in phonemes if p.isalpha() or p in [' ']]
        return ' '.join(phonemes)
    except Exception as e:
        print(f"Warning: Error converting '{text}': {e}")
        return text  # Fallback to original text

def process_ljspeech_metadata(dataset_path, output_path, train_split=0.95, seed=42):
    """
    Process LJSpeech metadata and convert to VITS format
    """
    
    print("=" * 60)
    print("LJSpeech to VITS Format Conversion")
    print("=" * 60)
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Paths
    metadata_file = os.path.join(dataset_path, "metadata.csv")
    wavs_dir = os.path.join(dataset_path, "wavs")
    
    # Check files exist
    if not os.path.exists(metadata_file):
        print(f"❌ Error: {metadata_file} not found!")
        return False
    
    if not os.path.exists(wavs_dir):
        print(f"❌ Error: {wavs_dir} not found!")
        return False
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Reading metadata from: {metadata_file}")
    print(f"Output directory: {output_path}\n")
    
    # Initialize g2p
    print("Initializing phoneme converter (g2p_en)...")
    g2p = G2p()
    print("✓ Phoneme converter ready\n")
    
    # Read metadata
    metadata = []
    with open(metadata_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        metadata = list(reader)
    
    print(f"Processing {len(metadata)} utterances...\n")
    
    # Process each utterance
    processed_data = []
    
    for i, row in enumerate(tqdm(metadata, desc="Converting to phonemes")):
        if len(row) < 3:
            print(f"Warning: Row {i} has only {len(row)} columns, skipping")
            continue
        filename = row[0]
        normalized_text = row[1]
        raw_text = row[2]
        
        # Check audio file exists
        wav_path = os.path.join(wavs_dir, f"{filename}.wav")
        if not os.path.exists(wav_path):
            continue
        
        # Convert to phonemes
        phonemes = text_to_phonemes(normalized_text, g2p)
        
        # Store data
        processed_data.append({
            'filename': filename,
            'text': normalized_text,
            'raw_text': raw_text,
            'phonemes': phonemes
        })
    
    print(f"\n✓ Successfully processed {len(processed_data)} utterances")
    
    # Split into train/validation
    num_train = int(len(processed_data) * train_split)
    
    train_data = processed_data[:num_train]
    val_data = processed_data[num_train:]
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_data)} ({100*len(train_data)/len(processed_data):.1f}%)")
    print(f"  Validation: {len(val_data)} ({100*len(val_data)/len(processed_data):.1f}%)")
    
    # Save train metadata
    train_file = os.path.join(output_path, "train.txt")
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            # Format: filename|phonemes
            f.write(f"{item['filename']}|{item['phonemes']}\n")
    
    # Save validation metadata
    val_file = os.path.join(output_path, "val.txt")
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(f"{item['filename']}|{item['phonemes']}\n")
    
    # Save detailed metadata (JSON for reference)
    metadata_json = os.path.join(output_path, "metadata.json")
    with open(metadata_json, 'w', encoding='utf-8') as f:
        json.dump({
            'total': len(processed_data),
            'train': len(train_data),
            'val': len(val_data),
            'sample': processed_data[:3]  # First 3 examples
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved training metadata to: {train_file}")
    print(f"✓ Saved validation metadata to: {val_file}")
    print(f"✓ Saved detailed metadata to: {metadata_json}")
    
    # Print sample
    print("\n" + "=" * 60)
    print("SAMPLE CONVERSIONS")
    print("=" * 60)
    
    for i in range(min(5, len(processed_data))):
        item = processed_data[i]
        print(f"\nFile: {item['filename']}")
        print(f"Text:     {item['text']}")
        print(f"Phonemes: {item['phonemes']}")
    
    print("\n" + "=" * 60)
    print("✓ Conversion complete!")
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
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        print(f"\nUsage: python prepare_data.py <dataset_path> <output_path>")
        print(f"Example: python prepare_data.py ./data/LJSpeech-1.1 ./data/ljspeech_prepared")
        sys.exit(1)
    
    # Process
    success = process_ljspeech_metadata(dataset_path, output_path)
    sys.exit(0 if success else 1)
