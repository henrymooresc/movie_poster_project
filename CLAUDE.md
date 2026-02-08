# CLAUDE.md

## Project Overview

Multi-label movie genre classification from poster images using deep learning. A FastAI/PyTorch project that fine-tunes ResNet50 to predict movie genres from poster artwork.

## Project Structure

```
movie_poster_project/
├── movie_poster_genre.ipynb   # Main notebook (data prep, training, inference)
├── MovieGenre.csv             # Dataset (~40K movies with poster URLs and genres)
├── data/                      # Downloaded poster images (~7,900 JPGs)
│   └── test/                  # Held-out test posters
└── .gitignore
```

## Tech Stack

- **Language:** Python 3.10
- **Deep Learning:** FastAI (built on PyTorch), ResNet50 transfer learning
- **Data:** Pandas, PIL/Pillow, Requests
- **Environment:** Conda/Mamba (`movieai` environment)

## Setup & Run

```bash
# Activate environment
conda activate movieai

# Install dependencies (no requirements.txt exists)
pip install fastai fastbook pandas pillow requests jupyter sentencepiece

# Run
jupyter notebook movie_poster_genre.ipynb
```

## How It Works

1. Load `MovieGenre.csv` → filter genres with 50+ occurrences → balance to 200 movies/genre
2. Download poster images from URLs to `data/`
3. Build FastAI DataBlock with MultiCategoryBlock for multi-label classification
4. Fine-tune ResNet50 for 4 epochs
5. Test on unseen posters in `data/test/` with top-3 genre predictions

## Key Details

- **No automated tests** — validation is done manually in notebook cells
- **No linting/formatting** configured
- **No CI/CD** pipeline
- **No requirements.txt** — dependencies are managed via the `movieai` conda environment
- **Remote:** `https://github.com/henrymooresc/movie_poster_project.git` (branch: `main`)
- Large files (`data/*.jpg`, `MovieGenre.csv`) are in the repo — be mindful of size
