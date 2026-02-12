# CLAUDE.md

## Project Overview

Multi-label movie genre classification from poster images using deep learning. A FastAI/PyTorch project that fine-tunes ResNet50 to predict movie genres from poster artwork, with a Gradio web interface for deployment on HuggingFace Spaces.

## Project Structure

```
movie_poster_project/
├── train_model.py                      # Training pipeline (data prep, model training, evaluation)
├── app_gradio.py                       # Gradio web interface for HuggingFace Spaces
├── movie_poster_genre.ipynb            # Original exploratory notebook (reference only)
├── requirements.txt                    # Python dependencies
├── MovieGenre.csv                      # Dataset (~40K movies with poster URLs and genres)
├── movie_genre_classifier_export.pkl   # Exported model for inference
├── movie_genre_classifier_vocab.txt    # Genre vocabulary list
├── README_HUGGINGFACE.md               # Model card / HuggingFace README
├── deploy_to_huggingface.md            # Step-by-step deployment guide
├── CLAUDE.md                           # This file
├── .gitignore
├── data/                               # Downloaded poster images (~7,900 JPGs)
└── models/                             # Saved model checkpoints
    ├── best_model.pth
    └── movie_genre_classifier.pth
```

## Tech Stack

- **Language:** Python 3.10
- **Deep Learning:** FastAI (built on PyTorch), ResNet50 transfer learning
- **Web Interface:** Gradio (for HuggingFace Spaces deployment)
- **Data:** Pandas, PIL/Pillow, Requests
- **Environment:** Conda/Mamba (`movieai` environment)

## Setup & Run

```bash
# Activate environment
conda activate movieai

# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Launch the Gradio web interface
python app_gradio.py
```

## Key Files

### train_model.py
Main training script with `MovieGenreTrainer` class. Handles the full pipeline:
1. Load `MovieGenre.csv` → filter genres with 50+ occurrences → balance to 200 movies/genre
2. Download poster images from URLs to `data/`
3. Build FastAI DataBlock with MultiCategoryBlock, 80/20 train/validation split
4. Fine-tune ResNet50 with mixed precision (FP16), custom poster augmentations
5. Evaluate with accuracy, F1, precision, and recall across multiple thresholds
6. Export model as `.pkl` for deployment

Three training strategies available (selectable in `main()`):
- **Standard fine-tuning** (default): `fine_tune(10)` with 3 frozen + 10 unfrozen epochs
- **Advanced discriminative LRs**: Different learning rates per layer group
- **Progressive resizing**: Train at 224px then 448px

### app_gradio.py
Gradio web interface with `GradioPredictor` class. Loads the exported `.pkl` model and serves predictions via a web UI. Supports adjustable top-k, confidence threshold, and optional TTA. Designed for deployment to HuggingFace Spaces.

### movie_poster_genre.ipynb
Original exploratory notebook. Kept as reference — `train_model.py` supersedes it for training.

## Key Details

- **No automated tests** — validation metrics are printed during training
- **No linting/formatting** configured
- **No CI/CD** pipeline
- **Remote:** `https://github.com/henrymooresc/movie_poster_project.git` (branch: `main`)
- Large files (`data/*.jpg`, `MovieGenre.csv`, `*.pkl`) are gitignored — only code and docs are tracked
- Custom multi-label metrics (`F1ScoreMulti`, `PrecisionMulti`, `RecallMulti`) are defined at the bottom of `train_model.py`
