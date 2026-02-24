# Movie Poster Genre Classifier

Multi-label movie genre classification from poster images using deep learning. This model predicts movie genres directly from poster artwork using a fine-tuned ResNet50 architecture.

## Model Description

- **Architecture**: ResNet50 (transfer learning)
- **Framework**: FastAI/PyTorch
- **Task**: Multi-label image classification
- **Training Data**: ~7,900 movie posters with genre labels
- **Genres**: Action, Adventure, Animation, Biography, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, News, Romance, Sci-Fi, Sport, Thriller, War, Western

## Usage

### Quick Start with Python

```python
from fastai.vision.all import *

# Load the model
learn = load_learner('movie_genre_classifier_export.pkl')

# Predict from image path
pred_class, pred_idxs, probs = learn.predict('poster.jpg')

# Get top 3 genres
top_probs, top_idxs = probs.topk(3)
top_genres = [learn.dls.vocab[i] for i in top_idxs]

for genre, prob in zip(top_genres, top_probs):
    print(f"{genre}: {prob:.1%}")
```

### Gradio Web Interface

```python
python app_gradio.py
```

Then open your browser to interact with the model via a web UI.

## Performance

The model uses several improvements for better accuracy:

- ✅ **Validation Split**: 80/20 train/validation split for proper evaluation
- ✅ **Optimized Training**: 10+ epochs with discriminative learning rates
- ✅ **Custom Augmentations**: Poster-specific augmentations (no flipping, minimal rotation)
- ✅ **Mixed Precision**: FP16 training for efficiency
- ✅ **Multiple Metrics**: Accuracy, F1, Precision, Recall tracking

### Key Features

- **Multi-label Prediction**: Can predict multiple genres per poster
- **Confidence Scores**: Returns probability for each genre
- **Threshold Tuning**: Customizable confidence threshold per genre
- **Test Time Augmentation**: Optional TTA for improved accuracy
- **Batch Processing**: Process multiple posters at once

## Training

To train your own model:

```bash
# Install dependencies
pip install -r requirements.txt

# Run training script
python train_model.py
```

Training options include:
- Standard fine-tuning
- Advanced discriminative learning rates
- Progressive resizing (224→448)

## Files

- `train_model.py` - Complete training pipeline with improvements
- `app_gradio.py` - Gradio web interface for the model
- `movie_genre_classifier_export.pkl` - Exported model for inference
- `movie_genre_classifier_vocab.txt` - Genre vocabulary

## Model Improvements

This model incorporates several best practices:

1. **Proper Validation Split** - 20% holdout for unbiased evaluation
2. **Extended Training** - 10+ epochs vs original 4 epochs
3. **Image Resolution** - Support for higher resolution (224→448)
4. **Better Augmentations** - Poster-optimized transforms
5. **Multiple Metrics** - Track accuracy, F1, precision, recall
6. **Threshold Optimization** - Per-genre threshold tuning
7. **Mixed Precision** - Faster training with FP16

## 📄 License

This model is released under the MIT License. The training data comes from IMDb posters and should be used in accordance with IMDb's terms of service.
