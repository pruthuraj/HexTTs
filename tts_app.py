"""
Simple VITS TTS Application
Interactive text-to-speech tool
"""

import os
import sys
import torch
import argparse
from pathlib import Path

from hextts.config import load_config
from hextts.inference import VITSInferencePipeline


class TTSApp:
    """Interactive TTS Application"""
    
    def __init__(self, checkpoint_path: str, config_path: str, device: str = 'cuda'):
        """
        Initialize TTS application
        
        Args:
            checkpoint_path: path to trained model
            config_path: path to config file
            device: cuda or cpu
        """

        # Load config from the shared loader
        self.config = load_config(config_path)
        
        # Set device
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        # Initialize inference engine
        self.inference = VITSInferencePipeline(checkpoint_path, self.config, self.device)
        
        # Output directory
        self.output_dir = Path('tts_output')
        self.output_dir.mkdir(exist_ok=True)
    
    def run_interactive(self):
        """Run interactive TTS mode"""
        print("\n" + "="*60)
        print("VITS Text-to-Speech Application")
        print("="*60)
        print("\nCommands:")
        print("  speak <text>  - Synthesize speech from text")
        print("  save <text>   - Synthesize and save to file")
        print("  help          - Show help")
        print("  exit/quit     - Exit application")
        print("="*60 + "\n")
        
        counter = 0
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                
                elif user_input.lower().startswith('speak '):
                    text = user_input[6:].strip()
                    if text:
                        audio, sr = self.inference.synthesize(text)
                        # Save temporarily
                        output_file = self.output_dir / f"tts_{counter}.wav"
                        self._save_audio(audio, sr, output_file)
                        counter += 1
                
                elif user_input.lower().startswith('save '):
                    text = user_input[5:].strip()
                    if text:
                        audio, sr = self.inference.synthesize(text)
                        filename = input("Enter filename (without .wav): ").strip()
                        if filename:
                            output_file = self.output_dir / f"{filename}.wav"
                            self._save_audio(audio, sr, output_file)
                
                else:
                    # Default: treat as text to speak
                    audio, sr = self.inference.synthesize(user_input)
                    output_file = self.output_dir / f"tts_{counter}.wav"
                    self._save_audio(audio, sr, output_file)
                    counter += 1
            
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    @staticmethod
    def _save_audio(audio, sr, output_file):
        """Save audio file"""
        try:
            import soundfile as sf
            sf.write(output_file, audio, sr)
            print(f"✓ Audio saved to: {output_file}")
        except ImportError:
            print("Error: soundfile not installed")
        except Exception as e:
            print(f"Error saving audio: {e}")
    
    @staticmethod
    def _show_help():
        """Show help message"""
        print("\n" + "="*60)
        print("HELP")
        print("="*60)
        print("""
Examples:

  > hello world
    Synthesizes "hello world" and saves as tts_0.wav
  
  > speak the quick brown fox
    Synthesizes with explicit "speak" command
  
  > save This is a longer sentence
    Synthesizes and asks for custom filename
  
  > help
    Shows this help message
  
  > exit
    Exits the application

Tips:
  - Text is converted to phonemes automatically
  - Audio files are saved to the 'tts_output' directory
  - Longer text takes longer to synthesize
  - The model was trained on English text
        """)
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='VITS Text-to-Speech Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  Interactive mode:
    python tts_app.py --checkpoint best_model.pt

  Command-line mode:
    python tts_app.py --checkpoint best_model.pt --text "hello world" --output hello.wav
        """
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                       help='Path to config file')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to synthesize (if not set, runs interactive mode)')
    parser.add_argument('--output', type=str, default='output.wav',
                       help='Output audio file (command-line mode)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Create app
    app = TTSApp(args.checkpoint, args.config, args.device)
    
    # Run mode
    if args.text:
        # Command-line mode
        print(f"\nSynthesizing: {args.text}")
        audio, sr = app.inference.synthesize(args.text)
        app._save_audio(audio, sr, Path(args.output))
    else:
        # Interactive mode
        app.run_interactive()


if __name__ == "__main__":
    main()
