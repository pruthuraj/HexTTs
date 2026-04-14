"""
Complete Setup Test for VITS TTS Project
Tests: PyTorch, CUDA, Audio libs, Phoneme conversion, Data loading
"""

import sys
import os


def test_pytorch():
    """Test PyTorch installation and CUDA availability"""
    print("\n" + "="*60)
    print("1. Testing PyTorch")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA available: YES")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            
            # Test GPU memory
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print(f"⚠ CUDA available: NO (CPU-only mode)")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_audio_libs():
    """Test audio processing libraries"""
    print("\n" + "="*60)
    print("2. Testing Audio Libraries")
    print("="*60)
    
    try:
        import librosa
        print(f"✓ librosa version: {librosa.__version__}")
        
        import soundfile
        print(f"✓ soundfile version: {soundfile.__version__}")
        
        import scipy
        print(f"✓ scipy version: {scipy.__version__}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_phoneme_conversion():
    """Test phoneme conversion"""
    print("\n" + "="*60)
    print("3. Testing Phoneme Conversion")
    print("="*60)
    
    try:
        from g2p_en import G2p
        print("✓ g2p_en imported successfully")
        
        g2p = G2p()
        
        # Small smoke set verifies end-to-end text->phoneme path.
        test_texts = [
            "Hello world",
            "The quick brown fox",
            "VITS text to speech"
        ]
        
        for text in test_texts:
            phonemes = g2p(text)
            phoneme_str = ''.join(phonemes)
            print(f"  '{text}' → {phoneme_str}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_numpy_version():
    """Test NumPy version (critical for compatibility)"""
    print("\n" + "="*60)
    print("4. Testing NumPy Version")
    print("="*60)
    
    try:
        import numpy as np
        version = np.__version__
        major = int(version.split('.')[0])
        
        print(f"✓ NumPy version: {version}")
        
        if major >= 2:
            print(f"⚠ WARNING: NumPy 2.x may cause compatibility issues")
            print(f"  Consider downgrading: pip install 'numpy<2'")
            return False
        else:
            print(f"✓ NumPy version is compatible (< 2.0)")
            return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_data_loading(data_path):
    """Test loading audio files"""
    print("\n" + "="*60)
    print("5. Testing Data Loading")
    print("="*60)
    
    try:
        import librosa
        import os
        
        # Find an audio file
        wavs_dir = os.path.join(data_path, "wavs")
        
        if not os.path.exists(wavs_dir):
            print(f"⚠ Dataset not found at: {wavs_dir}")
            print(f"  Skipping data loading test")
            return True  # Not a failure, just not present yet
        
        # Get first audio file
        wav_files = [f for f in os.listdir(wavs_dir) if f.endswith('.wav')]
        
        if not wav_files:
            print(f"⚠ No .wav files found in {wavs_dir}")
            return False
        
        # Load first file
        first_file = wav_files[0]
        wav_path = os.path.join(wavs_dir, first_file)
        
        audio, sr = librosa.load(wav_path, sr=22050)
        
        print(f"✓ Loaded test audio: {first_file}")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Duration: {len(audio)/sr:.2f} seconds")
        print(f"  Shape: {audio.shape}")
        
        # Compute one mel spectrogram to validate DSP stack works end-to-end.
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
        print(f"✓ Computed mel spectrogram")
        print(f"  Shape: {mel_spec.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_metadata_files(data_path):
    """Test that metadata files exist"""
    print("\n" + "="*60)
    print("6. Testing Prepared Metadata")
    print("="*60)
    
    try:
        train_file = os.path.join(data_path, "train.txt")
        val_file = os.path.join(data_path, "val.txt")
        
        if not os.path.exists(train_file):
            print(f"⚠ Training metadata not found: {train_file}")
            print(f"  Run: python scripts/prepare_data.py ./data/LJSpeech-1.1 ./data/ljspeech_prepared")
            return True
        
        # Count lines
        with open(train_file, 'r', encoding='utf-8') as f:
            train_count = sum(1 for _ in f)
        
        with open(val_file, 'r', encoding='utf-8') as f:
            val_count = sum(1 for _ in f)
        
        print(f"✓ Training metadata: {train_count} utterances")
        print(f"✓ Validation metadata: {val_count} utterances")
        
        # Show sample
        with open(train_file, 'r', encoding='utf-8') as f:
            sample = f.readline().strip()
        print(f"✓ Sample line: {sample[:80]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all setup checks and print a concise pass/fail summary."""
    print("\n" + "="*60)
    print("VITS TTS SETUP VERIFICATION")
    print("="*60)
    
    results = {
        'PyTorch': test_pytorch(),
        'Audio Libraries': test_audio_libs(),
        'Phoneme Conversion': test_phoneme_conversion(),
        'NumPy Version': test_numpy_version(),
    }
    
    # Check for dataset
    dataset_path = "./data/LJSpeech-1.1"
    prepared_path = "./data/ljspeech_prepared"
    
    results['Data Loading'] = test_data_loading(dataset_path)
    results['Prepared Metadata'] = test_metadata_files(prepared_path)
    
    # Summary section prints final setup readiness in one place.
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    
    all_pass = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_pass = False
    
    print("\n" + "="*60)
    
    if all_pass:
        print("✓ ALL TESTS PASSED!")
        print("\nYou're ready for Phase 3: Training Setup")
        print("Next: python setup_vits.py to configure VITS model")
    else:
        print("⚠ Some tests failed. Review the output above.")
        print("\nCommon issues:")
        print("  1. CUDA not available: Update NVIDIA drivers & install CUDA Toolkit")
        print("  2. NumPy 2.x: Run: pip install 'numpy<2'")
        print("  3. Dataset not found: Download LJSpeech to ./data/LJSpeech-1.1")
        print("  4. Metadata not prepared: Run: python scripts/prepare_data.py ...")
    
    print("="*60 + "\n")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
