"""
Main image processor that combines all processing steps.
"""

import os
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from PIL import Image
import piexif

from config import Config
from ollama_client import OllamaClient
from clip_processor import CLIPProcessor
from qdrant_manager import QdrantManager
from utils import (
    get_gps_coordinates,
    get_location_from_coordinates,
    extract_and_add_metadata,
    validate_file_path,
    _sanitize_for_json,
)


class ImageProcessor:
    """Main processor for handling image operations."""

    def __init__(self):
        """Initialize the image processor."""
        self.ollama_client = OllamaClient()
        self.clip_processor = CLIPProcessor()
        self.qdrant_manager = QdrantManager()

        # Initialize Qdrant collections
        success, message = self.qdrant_manager.create_collections()
        if not success:
            print(f"Warning: {message}")

        # Set allowed paths
        Config.set_allowed_paths(Config.get_allowed_paths())

    def check_system_status(self) -> Dict[str, Any]:
        """Check status of all system components."""
        status = {"timestamp": datetime.now().isoformat(), "components": {}}

        # Check Ollama client
        status["components"]["ollama"] = self.ollama_client.test_connection()

        # Check CLIP processor
        status["components"]["clip"] = self.clip_processor.test_connection()

        # Check Qdrant manager
        status["components"]["qdrant"] = self.qdrant_manager.test_connection()

        # Overall status
        all_ok = all(
            comp.get("status") in ["connected", "loaded"]
            for comp in status["components"].values()
        )
        status["overall_status"] = "healthy" if all_ok else "issues_detected"

        return status

    def process_single_image(
        self, image_path: str, force_reprocess: bool = False
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Process a single image through all steps.

        Args:
            image_path: Path to the image file
            force_reprocess: Whether to force reprocessing even if already processed

        Returns:
            Tuple of (result, error_message)
        """
        try:
            # Validate file path
            if not validate_file_path(image_path):
                return {"error": "File path not allowed"}, "File path not allowed"

            # Check if file exists
            if not os.path.exists(image_path):
                return {"error": "File not found"}, f"File not found: {image_path}"

            # Check if image is supported
            if not image_path.lower().endswith(tuple(Config.SUPPORTED_EXTENSIONS)):
                return {
                    "error": "Unsupported image format"
                }, f"Unsupported format: {image_path}"

            # Get image metadata
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    file_size = os.path.getsize(image_path)
                    format_name = img.format
            except Exception as e:
                return {"error": "Failed to read image"}, f"Image read error: {e}"

            # Extract GPS coordinates
            gps_coords = get_gps_coordinates(image_path)
            location_name = None
            if gps_coords:
                lat, lon = gps_coords
                location_name = get_location_from_coordinates(lat, lon)

            # Generate AI tags and description
            print(f"DEBUG: Starting AI analysis for image: {image_path}")
            ai_analysis, ai_error = self.ollama_client.generate_image_analysis(
                image_path
            )
            if ai_error:
                print(f"DEBUG: AI analysis failed: {ai_error}")
                return {"error": "AI analysis failed"}, ai_error
            print(f"DEBUG: AI analysis completed successfully")
            print(f"DEBUG: AI analysis result: {ai_analysis}")

            # Get image embedding
            clip_features, clip_error = self.clip_processor.get_image_features(
                image_path
            )
            if clip_error:
                return {"error": "CLIP processing failed"}, clip_error

            # Prepare image data for Qdrant
            image_data = {
                "image_id": str(uuid.uuid4()),
                "file_path": os.path.abspath(image_path),
                "file_name": os.path.basename(image_path),
                "file_size": file_size,
                "width": width,
                "height": height,
                "format": format_name,
                "processing_timestamp": datetime.now().isoformat(),
                "gps_coordinates": gps_coords,
                "location_name": location_name,
                "ai_tags": ai_analysis.get("tags", []),
                "ai_description": ai_analysis.get("description", ""),
                "model_used": ai_analysis.get("model_used", "unknown"),
                "embedding": clip_features.get("embedding", []),
                "embedding_dim": clip_features.get("embedding_dim", 0),
            }

            # Upsert to Qdrant
            success, error = self.qdrant_manager.upsert_image(
                image_data, image_data["embedding"]
            )
            if error:
                return {"error": "Database storage failed"}, error

            # Add metadata to image file
            try:
                metadata_to_add = {
                    "AI Description": image_data["ai_description"],
                    "AI Tags": ", ".join(image_data["ai_tags"]),
                    "Processing Date": image_data["processing_timestamp"],
                }

                if gps_coords:
                    metadata_to_add["GPS Latitude"] = str(gps_coords[0])
                    metadata_to_add["GPS Longitude"] = str(gps_coords[1])

                extract_and_add_metadata(image_path, metadata_to_add)
            except Exception as e:
                print(f"Warning: Failed to add metadata to image: {e}")

            # Return success result
            result = {
                "success": True,
                "image_data": image_data,
                "message": f"Successfully processed {os.path.basename(image_path)}",
            }

            return result, None

        except Exception as e:
            return {"error": "Processing failed"}, f"Error processing image: {e}"

    def process_bulk_images(
        self, directory_path: str, max_images: int = None
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Process multiple images in a directory.

        Args:
            directory_path: Path to directory containing images
            max_images: Maximum number of images to process

        Returns:
            Tuple of (result, error_message)
        """
        try:
            # Validate directory path
            if not validate_file_path(directory_path):
                return {
                    "error": "Directory path not allowed"
                }, "Directory path not allowed"

            if not os.path.isdir(directory_path):
                return {
                    "error": "Directory not found"
                }, f"Directory not found: {directory_path}"

            # Find image files
            image_files = []
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if file.lower().endswith(tuple(Config.SUPPORTED_EXTENSIONS)):
                        image_path = os.path.join(root, file)
                        image_files.append(image_path)

                # Limit number of images
                if max_images and len(image_files) >= max_images:
                    image_files = image_files[:max_images]
                    break

            if not image_files:
                return {
                    "error": "No images found"
                }, "No supported image files found in directory"

            # Process images
            results = {
                "total_images": len(image_files),
                "processed": 0,
                "failed": 0,
                "results": [],
                "start_time": datetime.now().isoformat(),
                "end_time": None,
            }

            for i, image_path in enumerate(image_files):
                print(
                    f"Processing image {i + 1}/{len(image_files)}: {os.path.basename(image_path)}"
                )

                try:
                    result, error = self.process_single_image(image_path)
                    if error:
                        results["failed"] += 1
                        results["results"].append(
                            {
                                "image_path": image_path,
                                "status": "failed",
                                "error": error,
                            }
                        )
                    else:
                        results["processed"] += 1
                        results["results"].append(
                            {
                                "image_path": image_path,
                                "status": "success",
                                "data": result.get("image_data", {}),
                            }
                        )
                except Exception as e:
                    results["failed"] += 1
                    results["results"].append(
                        {"image_path": image_path, "status": "failed", "error": str(e)}
                    )

            results["end_time"] = datetime.now().isoformat()

            return results, None

        except Exception as e:
            return {"error": "Bulk processing failed"}, f"Error in bulk processing: {e}"

    def search_images(
        self,
        query_embedding: List[float] = None,
        text_query: str = None,
        tags: List[str] = None,
        limit: int = 10,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Search for similar images.

        Args:
            query_embedding: Query embedding vector
            text_query: Text query for semantic search
            tags: List of tags to filter by
            limit: Maximum number of results

        Returns:
            Tuple of (results, error_message)
        """
        try:
            # Generate embedding from text query if provided
            if text_query and not query_embedding:
                query_embedding = self.clip_processor.get_text_embedding(
                    text_query
                ).tolist()

            if not query_embedding:
                return {
                    "error": "No query provided"
                }, "Please provide either text query or embedding"

            # Search for similar images
            results, error = self.qdrant_manager.search_similar_images(
                query_embedding=query_embedding, limit=limit
            )

            if error:
                return {"error": "Search failed"}, error

            # Filter by tags if provided
            if tags:
                filtered_results = []
                for result in results:
                    payload = result.get("payload", {})
                    result_tags = payload.get("ai_tags", [])

                    # Check if any of the requested tags are present
                    if any(
                        tag.lower() in [t.lower() for t in result_tags] for tag in tags
                    ):
                        filtered_results.append(result)

                results = filtered_results

            return {
                "query": text_query or "embedding",
                "results": results,
                "total_found": len(results),
                "limit": limit,
            }, None

        except Exception as e:
            return {"error": "Search failed"}, f"Error in search: {e}"

    def get_image_by_path(
        self, image_path: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Get image data by file path.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (image_data, error_message)
        """
        try:
            # Validate file path
            if not validate_file_path(image_path):
                return None, "File path not allowed"

            if not os.path.exists(image_path):
                return None, "File not found"

            # Search for the image in Qdrant
            # This is a simplified approach - in production you'd want a more efficient lookup
            all_collections = self.qdrant_manager.list_collections()

            for collection in all_collections:
                # This would need to be implemented properly in QdrantManager
                # For now, return None
                pass

            return None, "Image lookup not implemented"

        except Exception as e:
            return None, f"Error getting image: {e}"

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            "config": {
                "supported_extensions": Config.SUPPORTED_EXTENSIONS,
                "allowed_paths": Config.get_allowed_paths(),
                "max_image_size": Config.MAX_IMAGE_SIZE,
                "batch_size": Config.BATCH_SIZE,
            },
            "components": {
                "ollama": self.ollama_client.get_model_info(),
                "clip": self.clip_processor.get_model_info(),
                "qdrant": self.qdrant_manager.get_model_info(),
            },
            "timestamp": datetime.now().isoformat(),
        }
