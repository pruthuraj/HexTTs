import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf


def wav_to_mel(audio_path, sample_rate=22050, n_mels=80, n_fft=1024, hop_length=256, win_length=1024):
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=0,
        fmax=sr // 2,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db, sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to wav file")
    args = parser.parse_args()

    mel_db, sr = wav_to_mel(args.audio)

    plt.figure(figsize=(12, 5))
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", hop_length=256)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()