"""
CLIP processor for image vectorization using laion/clip-vit-b-32-laion2B-s34B-b79K.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, AutoModel
from sentence_transformers import SentenceTransformer

from config import Config
from utils import _sanitize_for_json


class CLIPProcessor:
    """Processor for CLIP image and text embeddings."""

    def __init__(self):
        """Initialize the CLIP processor."""
        self.model_name = Config.CLIP_MODEL_NAME
        self.model_path = Config.CLIP_MODEL_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model and processor
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Use local model path
                self.processor = AutoProcessor.from_pretrained(self.model_path)
                self.model = AutoModel.from_pretrained(self.model_path)
                print(f"Loaded CLIP model from local path: {self.model_path}")
            else:
                # Use remote model
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                print(f"Loaded CLIP model: {self.model_name}")

            self.model.to(self.device)
            self.model.eval()

            # Initialize sentence transformer for text embeddings
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

            self.model_info = Config.get_clip_model_info()
            self.model_info["device"] = str(self.device)

        except Exception as e:
            raise Exception(f"Failed to initialize CLIP model: {e}")

    def check_connection(self) -> tuple[bool, Optional[str]]:
        """Check if CLIP model is loaded and working."""
        try:
            # Test with a simple text
            test_text = "test"
            embeddings = self.get_text_embedding(test_text)
            if embeddings is not None and len(embeddings) > 0:
                return True, "CLIP model loaded successfully"
            else:
                return False, "CLIP model returned empty embeddings"
        except Exception as e:
            return False, f"CLIP model error: {e}"

    def get_image_embedding(self, image_path: str) -> tuple[np.ndarray, Optional[str]]:
        """
        Get image embedding from image file.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (embedding, error_message)
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")

            # Process image - CLIP processor needs both text and image inputs
            # We'll use a dummy text input since we only want image embeddings
            inputs = self.processor(
                text=["a photo"], images=image, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get image features
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_features = outputs.image_embeds

                # Normalize embeddings
                image_features = torch.nn.functional.normalize(
                    image_features, p=2, dim=1
                )
                embedding = image_features.cpu().numpy().flatten()

            return embedding, None

        except Exception as e:
            return np.array([]), f"Error getting image embedding: {e}"

    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get text embedding using sentence transformer.

        Args:
            text: Input text

        Returns:
            Text embedding as numpy array
        """
        try:
            # Use sentence transformer for text embeddings
            embedding = self.sentence_model.encode(text)
            return embedding
        except Exception as e:
            print(f"Error getting text embedding: {e}")
            return np.array([])

    def get_image_features(
        self, image_path: str
    ) -> tuple[Dict[str, Any], Optional[str]]:
        """
        Get comprehensive image features including embedding and metadata.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (features, error_message)
        """
        try:
            # Get image embedding
            embedding, error = self.get_image_embedding(image_path)
            if error:
                return {"error": error}, error

            # Get image dimensions
            with Image.open(image_path) as img:
                width, height = img.size

            # Create features dictionary
            features = {
                "embedding": embedding.tolist(),
                "embedding_dim": len(embedding),
                "image_path": image_path,
                "image_size": {"width": width, "height": height},
                "model_used": self.model_name,
                "timestamp": time.time(),
            }

            return features, None

        except Exception as e:
            return {"error": str(e)}, f"Error getting image features: {e}"

    def search_similar_images(
        self, query_embedding: np.ndarray, limit: int = 10
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Search for similar images based on embedding similarity.

        Args:
            query_embedding: Query embedding to search with
            limit: Maximum number of results

        Returns:
            Tuple of (results, error_message)
        """
        # This would typically search against a database of embeddings
        # For now, return empty list as placeholder
        return [], "Image search not implemented - requires database integration"

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": str(self.device),
            "embedding_dim": 512,  # CLIP-ViT-B-32 produces 512-dimensional embeddings
            "timestamp": time.time(),
        }

    def test_connection(self) -> Dict[str, Any]:
        """Test connection and get system status."""
        status = {
            "component": "clip",
            "status": "unknown",
            "model_info": {},
            "error": None,
        }

        # Check connection
        connected, message = self.check_connection()
        if connected:
            status["status"] = "loaded"
            status["model_info"] = self.get_model_info()
        else:
            status["status"] = "error"
            status["error"] = message

        return status

    def process_batch_images(
        self, image_paths: List[str]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Process multiple images in batch.

        Args:
            image_paths: List of image paths to process

        Returns:
            Tuple of (results, failed_paths)
        """
        results = []
        failed_paths = []

        for image_path in image_paths:
            try:
                features, error = self.get_image_features(image_path)
                if error:
                    failed_paths.append(image_path)
                    print(f"Failed to process {image_path}: {error}")
                else:
                    results.append(features)
            except Exception as e:
                failed_paths.append(image_path)
                print(f"Error processing {image_path}: {e}")

        return results, failed_paths
