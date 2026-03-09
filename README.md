# harmoni-ai - Music Genre Classifier

A convolutional neural network that classifies music into one of ten genres from raw audio. The model operates on log-scaled Mel spectrograms and is served through a FastAPI backend with a browser-based frontend.

---



https://github.com/user-attachments/assets/756a0d0d-a7e3-4f62-84e2-811cbe14999a



## Dataset

**GTZAN Genre Collection** — 1,000 audio tracks, 30 seconds each, 22,050 Hz, balanced across 10 genres.

| Genre | Samples |
|-------|---------|
| Blues | 100 |
| Classical | 100 |
| Country | 100 |
| Disco | 100 |
| Hip-Hop | 100 |
| Jazz | 100 |
| Metal | 100 |
| Pop | 100 |
| Reggae | 100 |
| Rock | 100 |

Split: **70% train / 15% validation / 15% test** (random seed 42)

---

## Preprocessing

Each 30-second track is cut into **4-second chunks with a 2-second overlap**. During training, one chunk is randomly sampled per track per epoch. During inference, all chunks are processed and their softmax scores averaged.

**Mel spectrogram settings:**

| Parameter | Value |
|-----------|-------|
| Sample rate | 22,050 Hz |
| n_fft | 1,024 |
| hop_length | 512 |
| n_mels | 128 |
| Log compression | log(x + 1e-9) |
| Resize target | 128 x 128 (bilinear) |

**Augmentation (training only):**
- SpecAugment: frequency masking (±15 bins) and time masking (±15 steps)
- Waveform-level perturbations applied before spectrogram computation

---

## Model Architecture

### ImprovedGenreCNN

A custom residual CNN with progressive channel expansion. Each residual block contains two 3x3 convolutions with batch normalization and a skip connection that uses a 1x1 conv for dimension matching when needed.

**Feature extractor:**

| Stage | Block | Out Channels | Spatial Op |
|-------|-------|-------------|------------|
| 1 | ResidualBlock(1 → 32) | 32 | MaxPool 2x2 |
| 2 | ResidualBlock(32 → 64) | 64 | MaxPool 2x2 |
| 3 | ResidualBlock(64 → 128) | 128 | MaxPool 2x2 |
| 4 | ResidualBlock(128 → 256) | 256 | MaxPool 2x2 + Dropout(0.3) |
| 5 | ResidualBlock(256 → 512) | 512 | AdaptiveAvgPool(1x1) |

**Classifier head:**
```
Linear(512, 256) → ReLU → Dropout(0.5) → Linear(256, 10)
```

Input shape: `(N, 1, 128, 128)` — Output: logits over 10 classes

### Baseline (Model 1)

An earlier flat CNN without residual connections trained in `model_1/`. Used to validate the preprocessing pipeline. Showed clear weaknesses on blues/jazz and country/rock pairs, which motivated the residual architecture.

---

## Experiments

Grid search over learning rate and batch size. Each configuration trained from scratch with early stopping (patience = 3 on validation loss) up to 40 epochs. Best checkpoint per config saved as TorchScript.

| Model File | Learning Rate | Batch Size |
|------------|-------------|------------|
| model_LR_0.001_BS_32.pt | 0.001 | 32 |
| model_LR_0.001_BS_64.pt | 0.001 | 64 |
| model_LR_0.0003_BS_32.pt | 0.0003 | 32 |
| model_LR_0.0003_BS_64.pt | 0.0003 | 64 |
| model_LR_0.0001_BS_32.pt | 0.0001 | 32 |
| model_LR_0.0001_BS_64.pt | 0.0001 | 64 |

Training curves logged to TensorBoard under `runs/music_genre_experiment`.

```bash
tensorboard --logdir runs/music_genre_experiment
```

---

## Evaluation

All six checkpoints evaluated on the held-out test set using accuracy, per-class precision/recall/F1, and a confusion matrix.

**Selected model: `model_LR_0.001_BS_64.pt`**

This configuration achieved the best validation accuracy while maintaining stable convergence. LR=0.001 with BS=64 reached a good optimum within the epoch budget without the validation loss diverging from training loss before early stopping.

**Harder genre pairs:**

| Pair | Reason |
|------|--------|
| Blues / Jazz | Overlapping harmonic structure and tempo |
| Country / Rock | Shared guitar timbre and rhythmic patterns |
| Disco / Pop | Similar production style and beat regularity |

---

## Inference

At inference the pipeline handles audio of any length:

1. Load audio, mix to mono, resample to 22,050 Hz
2. Truncate to 120 seconds maximum
3. Split into non-overlapping 4-second chunks, zero-pad the last chunk if needed
4. Compute 128x128 log Mel spectrogram for each chunk
5. Stack into a single batch tensor, run one forward pass
6. Average softmax probabilities across all chunks
7. Return the highest-probability genre + full confidence distribution

The uploaded file is deleted immediately after inference in a `finally` block regardless of success or failure.

---

## Project Structure

```
MUSIC_GENRE_CLASSIFIER/
├── backend/
│   ├── templates/
│   │   ├── index.html
│   │   ├── upload.html
│   │   └── result.html
│   ├── uploads/              # temp storage, auto-cleaned post-inference
│   ├── main.py               # FastAPI app
│   ├── loader.py             # TorchScript model loader
│   └── preprocessing.py      # chunking + spectrogram pipeline
├── model_1/
│   └── music_genre_model_1.pt
├── model_2/
│   ├── runs/                 # TensorBoard logs
│   ├── model_LR_0.001_BS_32.pt
│   ├── model_LR_0.001_BS_64.pt
│   ├── model_LR_0.0003_BS_32.pt
│   ├── model_LR_0.0003_BS_64.pt
│   ├── model_LR_0.0001_BS_32.pt
│   └── model_LR_0.0001_BS_64.pt
├── Dockerfile
├── requirements.txt
└── test_music_genre.ipynb
```

---

## Running Locally

```bash
pip install -r requirements.txt

export MODEL_PATH=model_2/model_LR_0.001_BS_64.pt

uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`

---

## Docker

The model is mounted as a volume rather than baked into the image, so you can swap checkpoints without rebuilding.

```bash
docker build -t harmoni-ai .

docker run -p 8000:8000 \
  -v $(pwd)/model_2/model_LR_0.001_BS_64.pt:/app/model_LR_0.001_BS_64.pt \
  -e MODEL_PATH=/app/model_LR_0.001_BS_64.pt \
  harmoni-ai
```
