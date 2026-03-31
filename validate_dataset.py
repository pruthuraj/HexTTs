"""
Dataset Validation Script for LJSpeech
Checks audio quality, sample rates, duration, and generates statistics
"""

import os
import csv
from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm

def validate_ljspeech_dataset(dataset_path):
    """
    Validate LJSpeech dataset integrity and quality
    """
    
    print("=" * 60)
    print("LJSpeech Dataset Validation")
    print("=" * 60)
    
    # Paths
    wavs_dir = os.path.join(dataset_path, "wavs")
    metadata_path = os.path.join(dataset_path, "metadata.csv")
    
    # Check existence
    if not os.path.exists(wavs_dir):
        print(f"❌ Error: {wavs_dir} not found!")
        return False
    
    if not os.path.exists(metadata_path):
        print(f"❌ Error: {metadata_path} not found!")
        return False
    
    print(f"✓ Dataset path: {dataset_path}")
    print()
    
    # Read metadata
    print("Reading metadata.csv...")
    metadata = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        metadata = list(reader)
    
    print(f"✓ Found {len(metadata)} utterances in metadata.csv")
    
    # Validate audio files
    print("\nValidating audio files...")
    
    stats = {
        'total_files': len(metadata),
        'valid_files': 0,
        'missing_files': 0,
        'sample_rates': {},
        'durations': [],
        'issues': []
    }
    
    for i, row in enumerate(tqdm(metadata[:100], desc="Checking files (first 100)")):
        filename = row[0]
        wav_path = os.path.join(wavs_dir, f"{filename}.wav")
        
        # Check if file exists
        if not os.path.exists(wav_path):
            stats['missing_files'] += 1
            stats['issues'].append(f"Missing: {filename}")
            continue
        
        try:
            # Load audio
            audio, sr = librosa.load(wav_path, sr=None)
            
            # Record sample rate
            if sr not in stats['sample_rates']:
                stats['sample_rates'][sr] = 0
            stats['sample_rates'][sr] += 1
            
            # Record duration
            duration = len(audio) / sr
            stats['durations'].append(duration)
            
            stats['valid_files'] += 1
            
        except Exception as e:
            stats['issues'].append(f"Error loading {filename}: {str(e)}")
    
    # Print results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    print(f"\nFile Statistics:")
    print(f"  Total utterances: {stats['total_files']}")
    print(f"  Valid audio files: {stats['valid_files']}")
    print(f"  Missing files: {stats['missing_files']}")
    
    print(f"\nSample Rates Found:")
    for sr, count in sorted(stats['sample_rates'].items()):
        print(f"  {sr} Hz: {count} files")
    
    if stats['durations']:
        durations = np.array(stats['durations'])
        print(f"\nAudio Duration Statistics:")
        print(f"  Min: {durations.min():.2f} seconds")
        print(f"  Max: {durations.max():.2f} seconds")
        print(f"  Mean: {durations.mean():.2f} seconds")
        print(f"  Median: {np.median(durations):.2f} seconds")
        print(f"  Total: {durations.sum() / 3600:.2f} hours")
    
    if stats['issues']:
        print(f"\n⚠ Issues Found ({len(stats['issues'])} total):")
        for issue in stats['issues'][:5]:  # Show first 5
            print(f"  - {issue}")
        if len(stats['issues']) > 5:
            print(f"  ... and {len(stats['issues']) - 5} more")
    else:
        print("\n✓ No issues found!")
    
    print("\n" + "=" * 60)
    
    return stats['valid_files'] > 0

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # Default path
        dataset_path = "./data/LJSpeech-1.1"
    
    # Check if path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Path not found: {dataset_path}")
        print(f"\nUsage: python validate_dataset.py <path_to_ljspeech>")
        print(f"Example: python validate_dataset.py ./data/LJSpeech-1.1")
        sys.exit(1)
    
    # Run validation
    success = validate_ljspeech_dataset(dataset_path)
    sys.exit(0 if success else 1)
