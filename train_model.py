"""
Movie Poster Genre Classification Training Script

Multi-label genre classification from movie poster images using FastAI/PyTorch.
Fine-tunes ResNet50/EfficientNet to predict movie genres from poster artwork.
"""

import pandas as pd
import requests
from pathlib import Path
from io import BytesIO
from PIL import Image
from fastai.vision.all import *
from fastai.callback.tracker import SaveModelCallback
import warnings
warnings.filterwarnings('ignore')


class MovieGenreTrainer:
    """Trainer class for movie genre classification model."""

    def __init__(self, csv_path='MovieGenre.csv', data_dir='data',
                 min_genre_count=50, samples_per_genre=200, seed=42):
        """
        Initialize the trainer.

        Args:
            csv_path: Path to MovieGenre.csv
            data_dir: Directory to save downloaded posters
            min_genre_count: Minimum occurrences for a genre to be included
            samples_per_genre: Number of samples per genre for balanced training
            seed: Random seed for reproducibility
        """
        self.csv_path = csv_path
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.min_genre_count = min_genre_count
        self.samples_per_genre = samples_per_genre
        self.seed = seed

        self.movie_df = None
        self.valid_sample_df = None
        self.dls = None
        self.learn = None

    def load_and_prepare_data(self):
        """Load CSV and prepare dataset with frequent genres."""
        print("📊 Loading and preparing data...")

        # Load CSV
        self.movie_df = pd.read_csv(
            self.csv_path,
            encoding='latin'
        ).dropna(subset=['Poster', 'Genre'])

        print(f"   Loaded {len(self.movie_df)} movies")

        # Get frequent genres
        genre_counts = self.movie_df['Genre'].str.split('|').explode().value_counts()
        frequent_genres = set(genre_counts[genre_counts >= self.min_genre_count].index)

        print(f"   Found {len(frequent_genres)} frequent genres: {sorted(frequent_genres)}")

        # Filter rare genres
        self.movie_df['Genre'] = self.movie_df['Genre'].apply(
            lambda x: self._filter_rare_genres(x, frequent_genres)
        )
        self.movie_df = self.movie_df.dropna(subset=['Genre'])

        print(f"   Filtered to {len(self.movie_df)} movies with frequent genres")

    def _filter_rare_genres(self, genre_str, frequent_genres):
        """Keep only frequent genres from a genre string."""
        if not isinstance(genre_str, str):
            return None

        genres = [g for g in genre_str.split('|') if g in frequent_genres]
        return '|'.join(genres) if genres else None

    def create_balanced_subset(self):
        """Create balanced subset of movies per genre."""
        print(f"\n🎯 Creating balanced subset ({self.samples_per_genre} per genre)...")

        # Get unique genres
        all_genres = set()
        for genres in self.movie_df['Genre'].str.split('|'):
            all_genres.update(genres)

        balanced_list = []
        for genre in all_genres:
            genre_subset_df = self.movie_df[self.movie_df['Genre'].str.contains(genre)]
            n = min(len(genre_subset_df), self.samples_per_genre)
            balanced_list.append(genre_subset_df.sample(n=n, random_state=self.seed))

        self.movie_subset_df = pd.concat(balanced_list).drop_duplicates(subset=['imdbId'])
        print(f"   Created subset with {len(self.movie_subset_df)} unique movies")

    def download_posters(self):
        """Download poster images for the subset."""
        print("\n📥 Downloading poster images...")

        valid_rows = []
        failed = 0

        for i, row in self.movie_subset_df.iterrows():
            try:
                path = self.data_dir / f"{row['imdbId']}.jpg"

                # Skip if already downloaded
                if not path.exists():
                    resp = requests.get(row['Poster'], timeout=5)
                    img = Image.open(BytesIO(resp.content)).convert('RGB')

                    # Validate image
                    if img.size[0] > 50 and img.size[1] > 50:
                        img.save(path)
                    else:
                        failed += 1
                        continue

                valid_rows.append(row)

            except Exception as e:
                failed += 1
                continue

        self.valid_sample_df = pd.DataFrame(valid_rows)
        print(f"   ✓ Downloaded {len(valid_rows)} posters ({failed} failed)")

    def create_dataloaders(self, img_size=224, batch_size=32, valid_pct=0.2):
        """
        Create FastAI DataLoaders with validation split.

        Args:
            img_size: Image size for training
            batch_size: Batch size
            valid_pct: Validation set percentage
        """
        print(f"\n🔧 Creating DataLoaders (image size: {img_size}x{img_size})...")

        # Custom augmentations optimized for posters
        batch_tfms = aug_transforms(
            mult=1.0,              # Keep aspect ratio
            do_flip=False,         # Don't flip posters
            max_rotate=5.0,        # Minimal rotation
            max_lighting=0.3,      # Moderate lighting changes
            max_warp=0.1,          # Minimal warping
            p_affine=0.5,
            p_lighting=0.75
        )

        dblock = DataBlock(
            blocks=(ImageBlock, MultiCategoryBlock),
            get_x=lambda r: self.data_dir / f"{r['imdbId']}.jpg",
            get_y=lambda r: r['Genre'].split('|'),
            splitter=RandomSplitter(valid_pct=valid_pct, seed=self.seed),  # Key improvement!
            item_tfms=Resize(img_size),
            batch_tfms=batch_tfms
        )

        self.dls = dblock.dataloaders(self.valid_sample_df, bs=batch_size)

        print(f"   Training set: {len(self.dls.train_ds)} samples")
        print(f"   Validation set: {len(self.dls.valid_ds)} samples")
        print(f"   Genres: {self.dls.vocab}")

    def create_learner(self, arch=resnet50, pretrained=True):
        """
        Create FastAI vision learner.

        Args:
            arch: Architecture (resnet50, resnet101, efficientnet_b3, etc.)
            pretrained: Use pretrained weights
        """
        print(f"\n🏗️  Creating learner with {arch.__name__}...")

        # Multiple metrics for better evaluation
        metrics = [
            partial(accuracy_multi, thresh=0.5),
            F1ScoreMulti(thresh=0.5),
            PrecisionMulti(thresh=0.5),
            RecallMulti(thresh=0.5)
        ]

        self.learn = vision_learner(
            self.dls,
            arch,
            metrics=metrics,
            pretrained=pretrained
        )

        # Use mixed precision for faster training
        self.learn = self.learn.to_fp16()

        print(f"   ✓ Learner created with mixed precision training")

    def find_learning_rate(self):
        """Find optimal learning rate."""
        print("\n📈 Finding optimal learning rate...")
        suggested_lr = self.learn.lr_find()
        print(f"   Suggested LR: {suggested_lr.valley:.2e}")
        return suggested_lr.valley

    def train(self, epochs=10, base_lr=1e-3, freeze_epochs=3):
        """
        Train the model with improved training strategy.

        Args:
            epochs: Total training epochs (after freeze)
            base_lr: Base learning rate
            freeze_epochs: Epochs to train with frozen backbone
        """
        print(f"\n🚀 Training model...")
        print(f"   Strategy: Freeze {freeze_epochs} epochs, then {epochs} epochs unfrozen")

        # Save best model callback
        save_callback = SaveModelCallback(
            monitor='valid_loss',
            fname='best_model'
        )

        # Fine-tune with discriminative learning rates
        self.learn.fine_tune(
            epochs,
            base_lr=base_lr,
            freeze_epochs=freeze_epochs,
            cbs=[save_callback]
        )

        print("\n✅ Training complete!")

    def train_advanced(self, freeze_epochs=3, unfreeze_epochs=10):
        """
        Advanced training with manual control and discriminative LRs.

        Args:
            freeze_epochs: Epochs with frozen backbone
            unfreeze_epochs: Epochs with unfrozen backbone
        """
        print(f"\n🚀 Advanced training strategy...")

        # Stage 1: Train head only
        print(f"\n   Stage 1: Training head ({freeze_epochs} epochs)...")
        self.learn.freeze()
        self.learn.fit_one_cycle(freeze_epochs, lr_max=1e-3)

        # Stage 2: Unfreeze and train with discriminative LRs
        print(f"\n   Stage 2: Fine-tuning all layers ({unfreeze_epochs} epochs)...")
        self.learn.unfreeze()
        self.learn.fit_one_cycle(
            unfreeze_epochs,
            lr_max=slice(1e-6, 1e-4)  # Lower LR for early layers
        )

        print("\n✅ Advanced training complete!")

    def progressive_resize(self, sizes=[224, 448], epochs_per_size=5):
        """
        Progressive resizing training strategy.

        Args:
            sizes: List of image sizes to train on
            epochs_per_size: Epochs to train at each size
        """
        print(f"\n📐 Progressive resizing training: {sizes}")

        for size in sizes:
            print(f"\n   Training at {size}x{size}...")

            # Recreate dataloaders with new size
            self.create_dataloaders(img_size=size)

            # Update learner's dataloaders
            if self.learn is not None:
                self.learn.dls = self.dls
            else:
                self.create_learner()

            # Train
            self.learn.fine_tune(epochs_per_size)

        print("\n✅ Progressive training complete!")

    def save_model(self, name='movie_genre_classifier'):
        """
        Save model for HuggingFace deployment.

        Args:
            name: Model name
        """
        print(f"\n💾 Saving model...")

        # Save FastAI model
        self.learn.save(name)
        print(f"   ✓ Saved FastAI model: {name}.pkl")

        # Export for inference
        self.learn.export(f'{name}_export.pkl')
        print(f"   ✓ Exported model: {name}_export.pkl")

        # Save vocabulary for reference
        vocab_path = Path(f'{name}_vocab.txt')
        vocab_path.write_text('\n'.join(self.dls.vocab))
        print(f"   ✓ Saved vocabulary: {vocab_path}")

        print(f"\n✅ Model ready for HuggingFace deployment!")

    def evaluate(self):
        """Evaluate model on validation set."""
        print("\n📊 Evaluating model on validation set...")

        # Get predictions
        preds, targs = self.learn.get_preds()

        # Calculate metrics at different thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

        print("\n   Threshold optimization:")
        for thresh in thresholds:
            acc = accuracy_multi(preds, targs, thresh=thresh).item()
            print(f"   Threshold {thresh:.1f}: Accuracy = {acc:.4f}")

        # Show interpretation
        interp = ClassificationInterpretation.from_learner(self.learn)
        print("\n   Top losses (most confused predictions):")
        interp.print_classification_report()


# Custom metrics for multi-label classification
class F1ScoreMulti(Metric):
    """F1 Score for multi-label classification."""
    def __init__(self, thresh=0.5):
        super().__init__()
        self.thresh = thresh

    def reset(self):
        self.tps, self.fps, self.fns = 0, 0, 0

    def accumulate(self, learn):
        preds = (learn.pred.sigmoid() > self.thresh).float()
        targs = learn.y.float()
        self.tps += (preds * targs).sum().item()
        self.fps += (preds * (1-targs)).sum().item()
        self.fns += ((1-preds) * targs).sum().item()

    @property
    def value(self):
        prec = self.tps / (self.tps + self.fps + 1e-8)
        rec = self.tps / (self.tps + self.fns + 1e-8)
        return 2 * prec * rec / (prec + rec + 1e-8)

    @property
    def name(self): return 'f1_multi'


class PrecisionMulti(Metric):
    """Precision for multi-label classification."""
    def __init__(self, thresh=0.5):
        super().__init__()
        self.thresh = thresh

    def reset(self):
        self.tps, self.fps = 0, 0

    def accumulate(self, learn):
        preds = (learn.pred.sigmoid() > self.thresh).float()
        targs = learn.y.float()
        self.tps += (preds * targs).sum().item()
        self.fps += (preds * (1-targs)).sum().item()

    @property
    def value(self):
        return self.tps / (self.tps + self.fps + 1e-8)

    @property
    def name(self): return 'precision_multi'


class RecallMulti(Metric):
    """Recall for multi-label classification."""
    def __init__(self, thresh=0.5):
        super().__init__()
        self.thresh = thresh

    def reset(self):
        self.tps, self.fns = 0, 0

    def accumulate(self, learn):
        preds = (learn.pred.sigmoid() > self.thresh).float()
        targs = learn.y.float()
        self.tps += (preds * targs).sum().item()
        self.fns += ((1-preds) * targs).sum().item()

    @property
    def value(self):
        return self.tps / (self.tps + self.fns + 1e-8)

    @property
    def name(self): return 'recall_multi'


def main():
    """Main training pipeline."""
    print("🎬 Movie Poster Genre Classification Training\n")
    print("=" * 60)

    # Initialize trainer
    trainer = MovieGenreTrainer(
        csv_path='MovieGenre.csv',
        data_dir='data',
        min_genre_count=50,
        samples_per_genre=200,
        seed=42
    )

    # Prepare data
    trainer.load_and_prepare_data()
    trainer.create_balanced_subset()
    trainer.download_posters()

    # Create dataloaders with validation split (KEY IMPROVEMENT!)
    trainer.create_dataloaders(
        img_size=224,      # Start with smaller size
        batch_size=32,
        valid_pct=0.2      # 20% validation split
    )

    # Create learner
    trainer.create_learner(arch=resnet50)

    # Find optimal learning rate
    # lr = trainer.find_learning_rate()  # Uncomment to find LR

    # Training Strategy Option 1: Standard fine-tuning (recommended for first run)
    print("\n" + "=" * 60)
    print("Training Strategy: Standard Fine-Tuning")
    print("=" * 60)
    trainer.train(epochs=10, base_lr=1e-3, freeze_epochs=3)

    # Training Strategy Option 2: Advanced with discriminative LRs (uncomment to use)
    # print("\n" + "=" * 60)
    # print("Training Strategy: Advanced Discriminative Learning Rates")
    # print("=" * 60)
    # trainer.train_advanced(freeze_epochs=3, unfreeze_epochs=10)

    # Training Strategy Option 3: Progressive resizing (uncomment to use)
    # print("\n" + "=" * 60)
    # print("Training Strategy: Progressive Resizing")
    # print("=" * 60)
    # trainer.progressive_resize(sizes=[224, 448], epochs_per_size=5)

    # Evaluate
    trainer.evaluate()

    # Save model
    trainer.save_model('movie_genre_classifier')

    print("\n" + "=" * 60)
    print("🎉 Training pipeline complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
