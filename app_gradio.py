"""
Gradio Web Interface for Movie Poster Genre Classification

Interactive web app for predicting movie genres from poster images.
Perfect for HuggingFace Spaces deployment.
"""

import gradio as gr
from fastai.vision.all import *
from pathlib import Path
import pandas as pd


class GradioPredictor:
    """Wrapper for model predictions in Gradio."""

    def __init__(self, model_path='movie_genre_classifier_export.pkl'):
        """Initialize with trained model."""
        try:
            print(f"Loading model from {model_path}...")
            self.learn = load_learner(model_path)
            print(f"✓ Model loaded successfully!")
            print(f"  Genres: {self.learn.dls.vocab}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def predict(self, image, top_k=5, use_tta=False):
        """
        Predict genres from image.

        Args:
            image: PIL Image or path
            top_k: Number of top predictions to return
            use_tta: Use Test Time Augmentation

        Returns:
            dict: Genre predictions with confidence scores
        """
        try:
            if use_tta:
                # TTA prediction
                preds, _ = self.learn.tta(dl=self.learn.dls.test_dl([image]))
                probs = preds[0]
            else:
                # Standard prediction
                _, _, probs = self.learn.predict(image)

            # Get top-k predictions
            top_probs, top_idxs = probs.topk(min(top_k, len(probs)))

            # Create results dictionary for Gradio
            results = {
                self.learn.dls.vocab[idx]: float(prob)
                for idx, prob in zip(top_idxs, top_probs)
            }

            return results

        except Exception as e:
            return {"Error": str(e)}

    def predict_with_details(self, image, top_k=5, threshold=0.5, use_tta=False):
        """
        Predict with detailed output including all genres above threshold.

        Returns:
            tuple: (top_predictions_dict, detailed_dataframe)
        """
        try:
            if use_tta:
                preds, _ = self.learn.tta(dl=self.learn.dls.test_dl([image]))
                probs = preds[0]
            else:
                _, _, probs = self.learn.predict(image)

            # Top-k predictions for label output
            top_probs, top_idxs = probs.topk(min(top_k, len(probs)))
            top_results = {
                self.learn.dls.vocab[idx]: float(prob)
                for idx, prob in zip(top_idxs, top_probs)
            }

            # All predictions for detailed table
            all_results = []
            for i, (genre, prob) in enumerate(zip(self.learn.dls.vocab, probs)):
                prob_val = float(prob)
                all_results.append({
                    'Genre': genre,
                    'Confidence': f"{prob_val:.1%}",
                    'Above Threshold': '✓' if prob_val >= threshold else '',
                    'Probability': prob_val
                })

            # Sort by probability
            all_results.sort(key=lambda x: x['Probability'], reverse=True)

            # Create DataFrame
            df = pd.DataFrame(all_results)
            df = df[['Genre', 'Confidence', 'Above Threshold']]  # Remove raw probability

            return top_results, df

        except Exception as e:
            return {"Error": str(e)}, pd.DataFrame({'Error': [str(e)]})


def create_interface():
    """Create Gradio interface."""

    # Initialize predictor
    predictor = GradioPredictor()

    # Define the prediction function for Gradio
    def predict_interface(image, top_k, threshold, use_tta):
        """Interface function for Gradio."""
        return predictor.predict_with_details(
            image,
            top_k=int(top_k),
            threshold=threshold,
            use_tta=use_tta
        )

    # Create interface
    demo = gr.Interface(
        fn=predict_interface,
        inputs=[
            gr.Image(type="pil", label="Upload Movie Poster"),
            gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Top K Predictions"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.5,
                step=0.05,
                label="Confidence Threshold"
            ),
            gr.Checkbox(
                label="Use Test Time Augmentation (slower, more accurate)",
                value=False
            )
        ],
        outputs=[
            gr.Label(num_top_classes=10, label="Top Predictions"),
            gr.Dataframe(label="All Genre Predictions")
        ],
        title="🎬 Movie Poster Genre Classifier",
        description="""
        **Predict movie genres from poster images!**

        Upload a movie poster and the AI will predict its genres based on visual elements.
        The model can identify multiple genres per poster (multi-label classification).

        **Supported Genres:**
        Action, Adventure, Animation, Biography, Comedy, Crime, Documentary, Drama,
        Family, Fantasy, History, Horror, Music, Mystery, News, Romance, Sci-Fi,
        Sport, Thriller, War, Western

        **Tips:**
        - Higher quality posters work best
        - Try adjusting the confidence threshold to see more/fewer genres
        - Enable TTA for potentially better accuracy (but slower)
        """,
        article="""
        ### About the Model

        This model uses a fine-tuned **ResNet50** architecture trained on ~7,900 movie posters.
        It employs several best practices for improved accuracy:

        - ✅ Proper train/validation split
        - ✅ Extended training (10+ epochs)
        - ✅ Custom augmentations optimized for posters
        - ✅ Mixed precision training
        - ✅ Multiple evaluation metrics

        **How it works:**
        1. Upload a movie poster image
        2. Model analyzes visual elements (colors, composition, faces, text)
        3. Predicts probabilities for each genre
        4. Returns top predictions with confidence scores

        **Built with:** [FastAI](https://www.fast.ai/) • PyTorch • Gradio

        ---

        Created by Henry Moore • [GitHub](https://github.com/henrymooresc/movie_poster_project)
        """,
        examples=[
            # Add example poster paths here if available
            # ["examples/poster1.jpg", 5, 0.5, False],
            # ["examples/poster2.jpg", 3, 0.4, False],
        ],
        theme=gr.themes.Soft(),
        flagging_mode="never"
    )

    return demo


def main():
    """Launch the Gradio app."""
    print("🚀 Launching Movie Poster Genre Classifier...")

    # Create and launch interface
    demo = create_interface()

    # Launch with share=True for public link (HuggingFace Spaces doesn't need this)
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,        # Default Gradio port
        share=False              # Set to True for temporary public link
    )


if __name__ == '__main__':
    main()
