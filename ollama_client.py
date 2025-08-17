"""
Ollama client for image tagging and description generation using OpenAI compatibility.
"""

import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from openai import OpenAI

from config import Config
from utils import _sanitize_for_json


class OllamaClient:
    """Client for interacting with Ollama using OpenAI compatibility."""

    def __init__(self):
        """Initialize the Ollama client."""
        self.client = OpenAI(
            base_url=Config.OLLAMA_BASE_URL,
            api_key=Config.OLLAMA_API_KEY,
        )
        self.model_name = Config.OLLAMA_MODEL
        self.model_info = Config.get_ollama_model_info()

    def check_connection(self) -> tuple[bool, Optional[str]]:
        """Check connection to Ollama server."""
        try:
            # Try to list models to check connection
            models = self.client.models.list()
            return True, "Connected to Ollama server"
        except Exception as e:
            return False, f"Connection failed: {e}"

    def generate_tags(
        self, image_path: str, max_tags: int = 10
    ) -> tuple[List[str], Optional[str]]:
        """
        Generate tags for an image using Ollama.

        Args:
            image_path: Path to the image file
            max_tags: Maximum number of tags to generate

        Returns:
            Tuple of (tags, error_message)
        """
        try:
            # Create a prompt for image tagging
            prompt = f"""
            Analyze the image at {image_path} and generate descriptive tags.
            Return exactly {max_tags} tags that describe the image content.
            Format the response as a JSON array of strings.
            Example: ["beach", "sunset", "ocean", "sky", "sand", "waves", "tropical", "scenic", "nature", "outdoor"]
            """

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
            )

            # Extract and parse the response
            content = response.choices[0].message.content.strip()

            try:
                # Try to parse as JSON first
                tags = json.loads(content)
                if isinstance(tags, list):
                    # Clean and filter tags
                    clean_tags = []
                    for tag in tags:
                        if isinstance(tag, str) and tag.strip():
                            clean_tag = tag.strip().lower()
                            if len(clean_tag) > 1:  # Filter out very short tags
                                clean_tags.append(clean_tag)

                    # Limit to max_tags
                    return clean_tags[:max_tags], None
            except json.JSONDecodeError:
                # If not JSON, try to extract tags from text
                lines = content.split("\n")
                tags = []
                for line in lines:
                    line = line.strip().strip("-").strip("*").strip()
                    if line and "," in line:
                        # Split by comma
                        sub_tags = [
                            tag.strip().lower()
                            for tag in line.split(",")
                            if tag.strip()
                        ]
                        tags.extend(sub_tags)
                    elif line:
                        # Add single tag
                        tags.append(line.lower())

                # Clean and limit tags
                clean_tags = [tag for tag in tags if len(tag) > 1]
                return clean_tags[:max_tags], None

            return [], "Failed to parse tags from response"

        except Exception as e:
            return [], f"Error generating tags: {e}"

    def generate_description(self, image_path: str) -> tuple[str, Optional[str]]:
        """
        Generate a description for an image using Ollama.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (description, error_message)
        """
        try:
            prompt = f"""
            Analyze the image at {image_path} and provide a detailed description.
            Describe the main subjects, setting, colors, mood, and any notable features.
            Write a coherent paragraph of 3-5 sentences.
            """

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7,
            )

            description = response.choices[0].message.content.strip()
            return description, None

        except Exception as e:
            return "", f"Error generating description: {e}"

    def generate_image_analysis(
        self, image_path: str
    ) -> tuple[Dict[str, Any], Optional[str]]:
        """
        Generate comprehensive image analysis including tags and description.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (analysis_result, error_message)
        """
        try:
            # Generate tags
            tags, error = self.generate_tags(image_path)
            if error:
                return {"error": error}, error

            # Generate description
            description, error = self.generate_description(image_path)
            if error:
                return {"error": error}, error

            # Create analysis result
            result = {
                "tags": tags,
                "description": description,
                "model_used": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "image_path": image_path,
            }

            return result, None

        except Exception as e:
            return {"error": str(e)}, f"Error generating image analysis: {e}"

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            model = self.client.models.retrieve(self.model_name)
            return {
                "model_id": model.id,
                "model_name": model.name,
                "created": model.created,
                "owned_by": model.owned_by,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "error": f"Failed to get model info: {e}",
                "model_name": self.model_name,
                "timestamp": datetime.now().isoformat(),
            }

    def list_models(self) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """List available models from Ollama."""
        try:
            models = self.client.models.list()
            model_list = []

            for model in models:
                model_info = {
                    "id": model.id,
                    "name": model.name,
                    "created": model.created,
                    "owned_by": model.owned_by,
                }
                model_list.append(model_info)

            return model_list, None

        except Exception as e:
            return [], f"Error listing models: {e}"

    def test_connection(self) -> Dict[str, Any]:
        """Test connection and get system status."""
        status = {
            "component": "ollama",
            "status": "unknown",
            "model_info": {},
            "error": None,
        }

        # Check connection
        connected, message = self.check_connection()
        if connected:
            status["status"] = "connected"
            status["model_info"] = self.get_model_info()
        else:
            status["status"] = "error"
            status["error"] = message

        return status
