import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def safe_float(x):
    """
    Convert numpy/scalar-like values to a plain Python float.
    If input has multiple values, return its mean as float.
    """
    return float(np.asarray(x).item()) if np.asarray(x).size == 1 else float(np.mean(x))

def spectral_flatness(audio: np.ndarray, n_fft: int = 1024, hop_length: int = 256) -> float:
    """Compute mean spectral flatness for a waveform."""
    if len(audio) == 0:
        return 0.0

    flatness = librosa.feature.spectral_flatness(
        y=audio.astype(np.float32),
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return float(np.mean(flatness))

def compute_silence_ratio(audio: np.ndarray, threshold: float = 0.01) -> float:
    """
    Compute fraction of samples with absolute amplitude below threshold.
    Returns 1.0 for empty audio.
    """
    if len(audio) == 0:
        return 1.0
    return float(np.mean(np.abs(audio) < threshold))


def evaluate_audio(audio_path: str, sr_target: int | None = None) -> dict:
    """
    Load audio, optionally resample, compute objective metrics,
    and return a report dictionary with heuristic verdicts.
    """
    # Read waveform and sample rate from file.
    audio, sr = sf.read(audio_path)

    # Convert multi-channel audio to mono by averaging channels.
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample if target sample rate is provided and differs.
    if sr_target is not None and sr != sr_target:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=sr_target)
        sr = sr_target

    # Basic waveform-level metrics.
    duration_sec = len(audio) / sr if sr > 0 else 0.0
    peak = np.max(np.abs(audio)) if len(audio) else 0.0
    rms = safe_float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    silence_ratio = compute_silence_ratio(audio, threshold=0.01)
    zcr = (
        safe_float(
            librosa.feature.zero_crossing_rate(
                audio, frame_length=1024, hop_length=256
            ).mean()
        )
        if len(audio)
        else 0.0
    )
    
    spectral_flatness_val = spectral_flatness(audio)

    # Mel-spectrogram based summary stats.
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32),
        sr=sr,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        fmin=0,
        fmax=8000,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_frames = mel.shape[1]
    mel_mean_db = safe_float(mel_db.mean())
    mel_std_db = safe_float(mel_db.std())

    # Rule-based diagnostic messages.
    verdicts = []

    if duration_sec < 0.7:
        verdicts.append("Too short: duration predictor likely still weak.")
    elif duration_sec < 1.2:
        verdicts.append("Short, but may still contain early speech structure.")
    else:
        verdicts.append("Duration is in a more realistic range.")

    if silence_ratio > 0.80:
        verdicts.append("Mostly silence or near-silence.")
    elif silence_ratio > 0.50:
        verdicts.append("Contains a lot of silence.")
    else:
        verdicts.append("Silence level looks acceptable.")

    if rms < 0.01:
        verdicts.append("Very low energy output.")
    elif rms < 0.03:
        verdicts.append("Low energy, likely weak/soft speech.")
    else:
        verdicts.append("Energy level looks reasonable.")

    if zcr > 0.25:
        verdicts.append("Possibly noisy/buzzy output.")
    else:
        verdicts.append("Waveform is not excessively noisy by ZCR.")
        
    flat = spectral_flatness_val

    if flat > 0.2:
        verdicts.append("Spectrum looks buzzy/noisy.")
    elif flat > 0.05:
        verdicts.append("Spectrum has mild noise present.")
    else:
        verdicts.append("Spectrum has more speech-like structure.")

    # Structured report output.
    return {
        "file": str(audio_path),
        "sample_rate": sr,
        "num_samples": len(audio),
        "duration_sec": round(duration_sec, 4),
        "peak_amplitude": round(float(peak), 6),
        "rms_energy": round(float(rms), 6),
        "silence_ratio": round(float(silence_ratio), 6),
        "zero_crossing_rate": round(float(zcr), 6),
        "spectral_flatness": round(float(spectral_flatness_val), 6),
        "mel_frames": int(mel_frames),
        "mel_mean_db": round(float(mel_mean_db), 4),
        "mel_std_db": round(float(mel_std_db), 4),
        "verdict": verdicts,
    }


def print_report(report: dict):
    """Pretty-print the evaluation report with visual indicators."""
    print("\n" + "="*60)
    print(" HexTTS Output Evaluation Report ")
    print("="*60)
    print(f"File               : {report['file']}")
    print(f"Sample rate        : {report['sample_rate']} Hz")
    print(f"Samples            : {report['num_samples']:,}")
    print(f"Duration           : {report['duration_sec']:>6.4f} s")
    print(f"Peak amplitude     : {report['peak_amplitude']:>6.6f}")
    print(f"RMS energy         : {report['rms_energy']:>6.6f}")
    print(f"Silence ratio      : {report['silence_ratio']:>6.6f}")
    print(f"Zero crossing rate : {report['zero_crossing_rate']:>6.6f}")
    print(f"Spectral flatness  : {report['spectral_flatness']:>6.6f}")
    print(f"Mel frames         : {report['mel_frames']:>6d}")
    print(f"Mel mean dB        : {report['mel_mean_db']:>6.4f} dB")
    print(f"Mel std dB         : {report['mel_std_db']:>6.4f} dB")
    print("\n Verdict:")
    for v in report["verdict"]:
        print(f"  • {v}")
    print("-"*60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate HexTTS generated wav file(s)")
    parser.add_argument(
        "--audio", 
        type=str, 
        default=".",
        help="Path to wav file or directory containing wav files (default: current directory)"
    )
    parser.add_argument(
        "--sample_rate", 
        type=int, 
        default=None, 
        help="Optional resample target (applies to all files)"
    )
    args = parser.parse_args()

    # Validate and resolve input path.
    input_path = Path(args.audio)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")

    # Collect audio files to evaluate
    audio_files = []
    
    if input_path.is_file():
        # Single file evaluation
        if input_path.suffix.lower() == '.wav':
            audio_files = [input_path]
        else:
            raise ValueError(f"Not a .wav file: {input_path}")
    
    elif input_path.is_dir():
        # Directory evaluation: find all .wav files
        audio_files = sorted(input_path.glob("*.wav"))
        if not audio_files:
            raise FileNotFoundError(f"No .wav files found in directory: {input_path}")
        print(f"Found {len(audio_files)} .wav file(s) to evaluate\n")
    
    # Run evaluation on all files
    all_reports = []
    for audio_path in audio_files:
        try:
            report = evaluate_audio(str(audio_path), sr_target=args.sample_rate)
            all_reports.append(report)
            print_report(report)
        except Exception as e:
            print(f"\n Error evaluating {audio_path}: {e}\n")
    
    # Summary if multiple files
    if len(all_reports) > 1:
        print("\n" + "="*60)
        print("BATCH EVALUATION SUMMARY")
        print("="*60)
        for i, report in enumerate(all_reports, 1):
            filename = Path(report['file']).name
            duration = report['duration_sec']
            rms = report['rms_energy']
            zcr = report['zero_crossing_rate']
            flatness = report['spectral_flatness']
            print(f"{i}. {filename:40s} | {duration:6.3f}s | RMS: {rms:.4f} | ZCR: {zcr:.4f} | Flat: {flatness:.4f}")


if __name__ == "__main__":
    main()