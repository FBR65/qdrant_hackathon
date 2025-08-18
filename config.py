"""
Configuration settings for the Qdrant Hackathon project.
"""

import os
from typing import List, Optional


class Config:
    """Configuration class for the application."""

    # Qdrant Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_TIMEOUT: int = 120
    COLLECTION_NAME: str = "image_db"

    # Ollama/OpenAI Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1/"
    OLLAMA_MODEL: str = "mistral-small3.2:latest"
    OLLAMA_API_KEY: str = "ollama"

    # CLIP Model Configuration
    CLIP_MODEL_NAME: str = "laion/clip-vit-b-32-laion2B-s34B-b79K"
    CLIP_MODEL_PATH: Optional[str] = None

    # Image Processing Configuration
    SUPPORTED_EXTENSIONS: List[str] = [
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
    ]

    # Gradio Configuration
    SERVER_NAME: str = "0.0.0.0"
    SERVER_PORT: int = 7860
    SHARE: bool = False

    # Allowed Paths Configuration
    ALLOWED_PATHS: List[str] = []

    # Processing Configuration
    BATCH_SIZE: int = 10
    MAX_IMAGE_SIZE: int = 1024  # Maximum image size for processing

    @classmethod
    def get_allowed_paths(cls) -> List[str]:
        """Get allowed paths for file access."""
        if not cls.ALLOWED_PATHS:
            # Default to user's home directory and current working directory
            home_dir = os.path.expanduser("~")
            current_dir = os.getcwd()
            return [home_dir, current_dir]
        return cls.ALLOWED_PATHS

    @classmethod
    def set_allowed_paths(cls, paths: List[str]) -> None:
        """Set allowed paths for file access."""
        cls.ALLOWED_PATHS = paths

    @classmethod
    def get_qdrant_collection_name(cls, distance: str = "cosine") -> str:
        """Get collection name with distance metric."""
        return f"{cls.COLLECTION_NAME}_{distance.lower()}"

    @classmethod
    def get_distance_metrics(cls) -> List[str]:
        """Get available distance metrics for Qdrant."""
        return ["cosine", "euclid", "dot", "manhattan"]

    @classmethod
    def get_ollama_model_info(cls) -> dict:
        """Get Ollama/OpenAI model information."""
        return {
            "model_name": cls.OLLAMA_MODEL,
            "base_url": cls.OLLAMA_BASE_URL,
            "api_key": cls.OLLAMA_API_KEY,
        }

    @classmethod
    def get_clip_model_info(cls) -> dict:
        """Get CLIP model information."""
        return {"model_name": cls.CLIP_MODEL_NAME, "model_path": cls.CLIP_MODEL_PATH}

    @classmethod
    def update_from_env(cls) -> None:
        """Update configuration from environment variables."""
        # Qdrant settings
        cls.QDRANT_HOST = os.getenv("QDRANT_HOST", cls.QDRANT_HOST)
        cls.QDRANT_PORT = int(os.getenv("QDRANT_PORT", cls.QDRANT_PORT))

        # Ollama/OpenAI settings
        cls.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", cls.OLLAMA_BASE_URL)
        cls.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", cls.OLLAMA_MODEL)
        cls.OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", cls.OLLAMA_API_KEY)

        # CLIP settings
        cls.CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", cls.CLIP_MODEL_NAME)
        cls.CLIP_MODEL_PATH = os.getenv("CLIP_MODEL_PATH", cls.CLIP_MODEL_PATH)

        # Server settings
        cls.SERVER_NAME = os.getenv("SERVER_NAME", cls.SERVER_NAME)
        cls.SERVER_PORT = int(os.getenv("SERVER_PORT", cls.SERVER_PORT))
        cls.SHARE = os.getenv("SHARE", "false").lower() == "true"

        # Allowed paths
        allowed_paths_env = os.getenv("ALLOWED_PATHS")
        if allowed_paths_env:
            cls.set_allowed_paths(allowed_paths_env.split(":"))

    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate the configuration and return a list of errors."""
        errors = []

        # Validate Qdrant settings
        if not cls.QDRANT_HOST:
            errors.append("QDRANT_HOST is required")

        if not isinstance(cls.QDRANT_PORT, int) or cls.QDRANT_PORT <= 0:
            errors.append("QDRANT_PORT must be a positive integer")

        # Validate Ollama/OpenAI settings
        if not cls.OLLAMA_MODEL:
            errors.append("OLLAMA_MODEL is required")

        if not cls.OLLAMA_BASE_URL:
            errors.append("OLLAMA_BASE_URL is required")

        # Validate CLIP settings
        if not cls.CLIP_MODEL_NAME:
            errors.append("CLIP_MODEL_NAME is required")

        # Validate server settings
        if not cls.SERVER_NAME:
            errors.append("SERVER_NAME is required")

        if not isinstance(cls.SERVER_PORT, int) or cls.SERVER_PORT <= 0:
            errors.append("SERVER_PORT must be a positive integer")

        # Validate allowed paths
        if not cls.ALLOWED_PATHS:
            errors.append("At least one allowed path is required")

        for path in cls.ALLOWED_PATHS:
            if not os.path.exists(path):
                errors.append(f"Allowed path does not exist: {path}")

        return errors
