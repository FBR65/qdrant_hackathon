"""
Ollama client for image tagging and description generation using OpenAI compatibility.
"""

import json
import os
import base64
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
            # DEBUG: Log the image path and check if file exists
            print(f"DEBUG: Attempting to process image at path: {image_path}")
            print(f"DEBUG: File exists: {os.path.exists(image_path)}")

            if not os.path.exists(image_path):
                return [], f"Image file not found: {image_path}"

            # Check file size
            file_size = os.path.getsize(image_path)
            print(f"DEBUG: Image file size: {file_size} bytes")

            # Read and encode image as base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            print(
                f"DEBUG: Image encoded to base64, length: {len(base64_image)} characters"
            )

            # Create a prompt for image tagging
            prompt = f"""
            Analysiere das bereitgestellte Bild und generiere beschreibende Tags.
            Erstelle genau {max_tags} Tags, die den Bildinhalt beschreiben.
            Formatiere die Antwort als JSON-Array von Strings.
            Gib die Tags auf Deutsch aus.
            Beispiel: ["strand", "sonnenuntergang", "ozean", "himmel", "sand", "wellen", "tropisch", "landschaft", "natur", "draußen"]
            """

            print(f"DEBUG: Sending image and prompt to model: {self.model_name}")
            print(f"DEBUG: Prompt length: {len(prompt)} characters")

            # Create message with image content
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[message],
                max_tokens=500,
                temperature=0.3,
            )

            print(f"DEBUG: API call successful, response received")

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
            # DEBUG: Log the image path for description generation
            print(f"DEBUG: Generating description for image: {image_path}")

            # Read and encode image as base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            print(
                f"DEBUG: Image encoded to base64 for description, length: {len(base64_image)} characters"
            )

            prompt = f"""
            Analysiere das bereitgestellte Bild und gib eine detaillierte Beschreibung.
            Beschreibe die Hauptmotive, die Umgebung, Farben, Stimmung und alle bemerkenswerten Merkmale.
            Schreibe einen zusammenhängenden Absatz von 3-5 Sätzen auf Deutsch.
            """

            print(
                f"DEBUG: Sending image and description prompt to model: {self.model_name}"
            )

            # Create message with image content
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[message],
                max_tokens=300,
                temperature=0.7,
            )

            print(f"DEBUG: Description API call successful")

            description = response.choices[0].message.content.strip()
            print(f"DEBUG: Generated description length: {len(description)} characters")
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
            print(f"DEBUG: Starting comprehensive image analysis for: {image_path}")

            # Generate tags
            print(f"DEBUG: Step 1 - Generating tags")
            tags, error = self.generate_tags(image_path)
            if error:
                print(f"DEBUG: Tag generation failed: {error}")
                return {"error": error}, error
            print(f"DEBUG: Generated {len(tags)} tags: {tags}")

            # Generate description
            print(f"DEBUG: Step 2 - Generating description")
            description, error = self.generate_description(image_path)
            if error:
                print(f"DEBUG: Description generation failed: {error}")
                return {"error": error}, error
            print(f"DEBUG: Description generated successfully")

            # Create analysis result
            result = {
                "tags": tags,
                "description": description,
                "model_used": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "image_path": image_path,
            }

            print(f"DEBUG: Image analysis completed successfully")
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
