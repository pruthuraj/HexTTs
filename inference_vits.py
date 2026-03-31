"""
Inference Script for VITS TTS
Generates speech from text using trained model
"""

import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import argparse
from typing import Tuple, Optional
import yaml

from vits_model import VITS, VOCAB_SIZE
from vits_data import PHONEME_TO_ID


class VITSInference:
    """Inference class for VITS model"""
    
    def __init__(self, checkpoint_path: str, config: dict, device: torch.device):
        """
        Initialize inference
        
        Args:
            checkpoint_path: path to trained model checkpoint
            config: configuration dict
            device: torch device
        """
        self.device = device
        self.config = config
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        config['vocab_size'] = VOCAB_SIZE
        self.model = VITS(config).to(device)
        self.model.eval()
        
        # Load checkpoint
        # checkpoint = torch.load(checkpoint_path, map_location=device)
        # Use weights_only=False to load optimizer state if needed, but we only need model state dict here
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print("Model loaded successfully!")
        
        # Audio parameters
        self.sample_rate = config['sample_rate']
        self.n_mel_channels = config['n_mel_channels']
        self.mel_n_fft = config['mel_n_fft']
        self.mel_hop_length = config['mel_hop_length']
        self.mel_win_length = config['mel_win_length']
        self.mel_f_min = config['mel_f_min']
        self.mel_f_max = config['mel_f_max']
        self.ref_level_db = config['ref_level_db']
        self.min_level_db = config['min_level_db']
    
    def text_to_phonemes(self, text: str) -> str:
        """
        Convert text to phoneme sequence using g2p_en
        
        Args:
            text: input text
        
        Returns:
            phoneme string (space-separated)
        """
        try:
            from g2p_en import G2p
            g2p = G2p()
            phoneme_list = g2p(text)
            
            # Remove stress markers and filter
            phonemes = [p.rstrip('012') for p in phoneme_list]
            phonemes = [p for p in phonemes if p and p != ' ']
            
            return ' '.join(phonemes)
        except Exception as e:
            print(f"Error converting text to phonemes: {e}")
            return text
    
    def phonemes_to_ids(self, phoneme_str: str) -> torch.Tensor:
        """
        Convert phoneme string to ID tensor
        
        Args:
            phoneme_str: space-separated phonemes
        
        Returns:
            tensor of shape (1, seq_len)
        """
        phonemes = phoneme_str.split()
        ids = []
        
        for p in phonemes:
            p = p.strip().upper()
            if p in PHONEME_TO_ID:
                ids.append(PHONEME_TO_ID[p])
            else:
                print(f"Warning: Unknown phoneme '{p}', skipping")
        
        if not ids:
            # Return silence if no valid phonemes
            ids = [PHONEME_TO_ID['PAD']]
        
        return torch.LongTensor(ids).unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def generate_mel_spectrogram(self, phoneme_ids: torch.Tensor) -> np.ndarray:
        """
        Generate mel-spectrogram from phoneme IDs
        
        Args:
            phoneme_ids: tensor of shape (1, seq_len)
        
        Returns:
            mel-spectrogram of shape (n_mel_channels, time_steps)
        """
        
        print("Entering model forward...")
        outputs = self.model(phoneme_ids)
        print("Model forward completed.")

        # Check if 'predicted_mel' is in outputs
        if 'predicted_mel' not in outputs:
            raise ValueError("Model output missing 'predicted_mel'")

        mel_spec = outputs['predicted_mel']


        # Check for NaN or Inf values in the predicted mel spectrogram
        if torch.isnan(mel_spec).any() or torch.isinf(mel_spec).any():
            raise ValueError("predicted_mel contains NaN or Inf")

        print(f"Raw predicted mel tensor shape: {mel_spec.shape}")

        # Remove batch dimension and move to CPU
        mel_spec = mel_spec.squeeze(0).cpu().numpy()
        print(f"Mel after squeeze shape: {mel_spec.shape}")
        # Convert from log scale to dB
        mel_spec = mel_spec * -self.ref_level_db + self.ref_level_db

        return mel_spec
    
    def mel_spectrogram_to_audio(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Convert mel-spectrogram to audio using Griffin-Lim
        
        Args:
            mel_spec: mel-spectrogram of shape (n_mel_channels, time_steps)
        
        Returns:
            audio waveform
        """
        # Convert from dB to power
        mel_spec = np.power(10.0, mel_spec / 10.0)
        
        # Invert mel spectrogram
        spectrogram = librosa.feature.inverse.mel_to_stft(
            mel_spec,
            sr=self.sample_rate,
            n_fft=self.mel_n_fft,
            # Use the same fmin and fmax as during training for consistency
            fmin=self.mel_f_min,
            fmax=self.mel_f_max
        )
        
        # Griffin-Lim to convert spectrogram to waveform
        audio = librosa.griffinlim(
            spectrogram,
            n_iter=100,
            hop_length=self.mel_hop_length,
            win_length=self.mel_win_length
        )
        
        # Normalize
        audio = audio / (np.abs(audio).max() + 1e-7)
        
        return audio
    
    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text
        
        Args:
            text: input text
        
        Returns:
            (audio, sample_rate)
        """
        print(f"Input text: {text}")
        
        # Text → Phonemes
        phoneme_str = self.text_to_phonemes(text)
        print(f"Phonemes: {phoneme_str}")
        
        # Phonemes → IDs
        phoneme_ids = self.phonemes_to_ids(phoneme_str)
        print(f"Phoneme IDs shape: {phoneme_ids.shape}")
        
        # Mel-spectrogram generation
        try:
            mel_spec = self.generate_mel_spectrogram(phoneme_ids)
            print(f"Mel-spectrogram shape: {mel_spec.shape}")
        except Exception as e:
            print(f"Error during mel generation: {e}")
            raise

        # Mel-spectrogram → Audio
        try:
            audio = self.mel_spectrogram_to_audio(mel_spec)
            print(f"Audio shape: {audio.shape}, duration: {len(audio) / self.sample_rate:.2f}s")
        except Exception as e:
            print(f"Error during audio reconstruction: {e}")
            raise

        return audio, self.sample_rate


def main():
    parser = argparse.ArgumentParser(description='VITS TTS Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='vits_config.yaml',
                       help='Path to config file')
    parser.add_argument('--text', type=str, required=True,
                       help='Text to synthesize')
    parser.add_argument('--output', type=str, default='output.wav',
                       help='Output audio file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Initialize inference
    inference = VITSInference(args.checkpoint, config, device)
    
    # Synthesize
    audio, sr = inference.synthesize(args.text)
    
    # Save audio
    sf.write(args.output, audio, sr)
    print(f"\nAudio saved to {args.output}")


if __name__ == "__main__":
    main()
