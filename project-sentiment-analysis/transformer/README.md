# Movie Review Sentiment (Transformer)

5-class sentiment classification on the Rotten Tomatoes phrases dataset using Hugging Face transformer encoders. The script fine-tunes a checkpoint, evaluates on a held-out split, and produces a `result.csv` submission.

**Best observed accuracy**: **70.378%** (twitter-roberta-base-sentiment-latest, 2 epochs, cosine scheduler, warmup_ratio=0.1).

---

## Environment

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8 cores+ |
| Memory | 8GB | 16GB+ |
| GPU | Optional | CUDA GPU for 5-10x faster training |
| Storage | 3GB | 6GB+ (checkpoints and caches) |

### Software

- Python >= 3.10
- PyTorch >= 2.0.0
- transformers >= 4.38
- datasets >= 2.17
- scikit-learn >= 1.3.0
- pandas >= 2.0.0, numpy >= 1.24.0

### Python dependencies

```bash
torch>=2.0.0
transformers>=4.38
datasets>=2.17
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
```

---

## Installation

### Option 1: uv (recommended)

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Activate the virtualenv
source .venv/bin/activate   # Linux/macOS
# or
.venv\Scripts\activate      # Windows
```

### Option 2: pip

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS

pip install "torch>=2.0.0" "transformers>=4.38" "datasets>=2.17" \
            "scikit-learn>=1.3.0" "pandas>=2.0.0" "numpy>=1.24.0"
```

### Quick check

```bash
python3 - <<'PY'
import torch, transformers, datasets
print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)
PY
```

---

## Project layout

```
transformer/
├── run_sentiment.py      # Fine-tune a model, evaluate, and write result.csv
├── demo_inference.py     # Interactive CLI demo with a saved checkpoint
├── analysis.md           # Experiment notes and hyperparameter sweeps
└── ../local-datasets/    # train.tsv, test.tsv, sampleSubmission.csv (shared)
```

`run_sentiment.py` saves checkpoints and logs to `project-sentiment-analysis/model-output` by default and writes the submission to `project-sentiment-analysis/result.csv`.

---

## Usage

### 1) Fine-tune and generate submission

```bash
python3 run_sentiment.py \
  --data_dir project-sentiment-analysis/local-datasets \
  --model_name twitter-roberta-base-sentiment-latest \
  --num_train_epochs 2 \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --output_dir project-sentiment-analysis/model-output \
  --result_path project-sentiment-analysis/result.csv
```

Key flags:
- `--model_name`: Any HF checkpoint (distilroberta-base, distil-bert-base-cased, twitter-roberta-base-sentiment-latest, etc.).
- `--eval_fraction`: Portion of training data for validation (default 0.1).
- `--freeze_base`: If set, trains only the classifier head.
- `--ignore_mismatched_sizes`: Keeps loading even if the head shape differs from num_labels (default True).

Artifacts:
- Best checkpoint and tokenizer in `output_dir`.
- Validation metrics printed after training.
- `result.csv` submission aligned to `sampleSubmission.csv`.

### 2) Interactive inference

After fine-tuning, launch a prompt-driven demo:

```bash
python3 demo_inference.py --model_dir project-sentiment-analysis/model-output
```

Enter any phrase to get the predicted label (0-4) and confidence. Type `quit` to exit.

---

## Data

- `train.tsv`: PhraseId, SentenceId, Phrase, Sentiment (labels 0-4).
- `test.tsv`: PhraseId, SentenceId, Phrase (no labels).
- `sampleSubmission.csv`: Output schema example.

`run_sentiment.py` renames `Phrase` -> `text` and `Sentiment` -> `label`, then stratifies a validation split via `--eval_fraction`.

---

## Tips and notes

- GPU strongly recommended; CPU runs are slower but functional.
- Cosine schedule with `warmup_ratio=0.1` worked well in experiments.
- Full-parameter tuning outperformed head-only training for this task.
- To try smaller or faster runs, lower `--max_length` via tokenizer kwargs by editing the script or reduce epochs/batch size.
