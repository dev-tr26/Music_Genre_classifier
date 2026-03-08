import librosa
import torch
import torch.nn.functional as F
import torchaudio

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=44100,
    n_mels=128
)

def preprocess_long_audio(file_path):

    waveform, sr = librosa.load(file_path, sr=44100)

    waveform = torch.tensor(waveform)

    chunk_samples = 4 * 44100
    hop_samples = 2 * 44100

    chunks = []

    if waveform.shape[0] < chunk_samples:
        pad = chunk_samples - waveform.shape[0]
        waveform = F.pad(waveform, (0, pad))

    for start in range(0, waveform.shape[0] - chunk_samples + 1, hop_samples):

        chunk = waveform[start:start + chunk_samples]

        mel = mel_transform(chunk)

        mel = torch.log(mel + 1e-9)

        mel = mel.unsqueeze(0).unsqueeze(0)

        mel = F.interpolate(
            mel,
            size=(150,150),
            mode="bilinear",
            align_corners=False
        )

        mel = mel.squeeze(0)

        chunks.append(mel)

    return torch.stack(chunks)