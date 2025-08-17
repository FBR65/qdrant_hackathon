"""
Qdrant manager for vector database operations.
"""

import os
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from config import Config
from utils import _sanitize_for_json, create_payload


class QdrantManager:
    """Manager for Qdrant vector database operations."""

    def __init__(self):
        """Initialize the Qdrant manager."""
        self.host = Config.QDRANT_HOST
        self.port = Config.QDRANT_PORT
        self.timeout = Config.QDRANT_TIMEOUT
        self.collection_name = Config.COLLECTION_NAME
        self.distance_metrics = Config.get_distance_metrics()

        # Initialize Qdrant client
        try:
            self.client = QdrantClient(
                host=self.host, port=self.port, timeout=self.timeout
            )
            print(f"Connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            raise Exception(f"Failed to connect to Qdrant: {e}")

        # Initialize collections
        self.collections_created = False

    def check_connection(self) -> tuple[bool, Optional[str]]:
        """Check connection to Qdrant server."""
        try:
            # Try to get server info
            info = self.client.get_fast()
            return True, f"Connected to Qdrant server version {info.version}"
        except Exception as e:
            return False, f"Connection failed: {e}"

    def create_collections(self) -> tuple[bool, Optional[str]]:
        """Create collections for different distance metrics."""
        if self.collections_created:
            return True, "Collections already created"

        try:
            for distance in self.distance_metrics:
                collection_name = Config.get_qdrant_collection_name(distance)

                # Check if collection exists
                if self.client.collection_exists(collection_name=collection_name):
                    print(f"Collection {collection_name} already exists")
                    continue

                # Create collection
                distance_map = {
                    "cosine": models.Distance.COSINE,
                    "euclid": models.Distance.EUCLID,
                    "dot": models.Distance.DOT,
                    "manhattan": models.Distance.MANHATTAN,
                }

                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=512,  # CLIP-ViT-B-32 produces 512-dimensional embeddings
                        distance=distance_map.get(distance, models.Distance.COSINE),
                    ),
                )
                print(f"Created collection: {collection_name}")

            self.collections_created = True
            return True, "Collections created successfully"

        except Exception as e:
            return False, f"Failed to create collections: {e}"

    def upsert_image(
        self, image_data: Dict[str, Any], embedding: List[float]
    ) -> tuple[bool, Optional[str]]:
        """
        Upsert image data into Qdrant collections.

        Args:
            image_data: Dictionary containing image metadata
            embedding: Image embedding vector

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Generate unique ID for the image
            point_id = str(uuid.uuid4())

            # Create payload
            payload = create_payload(image_data)

            # Upsert into all collections
            for distance in self.distance_metrics:
                collection_name = Config.get_qdrant_collection_name(distance)

                self.client.upsert(
                    collection_name=collection_name,
                    points=[
                        models.PointStruct(
                            id=point_id, vector=embedding, payload=payload
                        )
                    ],
                )

            return True, None

        except Exception as e:
            return False, f"Failed to upsert image: {e}"

    def search_similar_images(
        self,
        query_embedding: List[float],
        limit: int = 10,
        distance_metric: str = "cosine",
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Search for similar images based on embedding similarity.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            distance_metric: Distance metric to use for search

        Returns:
            Tuple of (results, error_message)
        """
        try:
            collection_name = Config.get_qdrant_collection_name(distance_metric)

            # Search for similar images
            search_result = self.client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                search_params=models.SearchParams(hnsw_ef=256, exact=True),
                limit=limit,
            )

            # Process results
            results = []
            for point in search_result.points:
                result = {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload,
                    "distance": point.score if distance_metric == "cosine" else None,
                }
                results.append(result)

            return results, None

        except Exception as e:
            return [], f"Search failed: {e}"

    def get_image_by_id(
        self, image_id: str, distance_metric: str = "cosine"
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Get image data by ID.

        Args:
            image_id: Image ID to retrieve
            distance_metric: Distance metric to use

        Returns:
            Tuple of (image_data, error_message)
        """
        try:
            collection_name = Config.get_qdrant_collection_name(distance_metric)

            # Get image data
            result = self.client.retrieve(
                collection_name=collection_name, ids=[image_id]
            )

            if result:
                return {
                    "id": result[0].id,
                    "payload": result[0].payload,
                    "vector": result[0].vector,
                }, None
            else:
                return None, "Image not found"

        except Exception as e:
            return None, f"Failed to retrieve image: {e}"

    def delete_image(self, image_id: str) -> tuple[bool, Optional[str]]:
        """
        Delete image from all collections.

        Args:
            image_id: Image ID to delete

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Delete from all collections
            for distance in self.distance_metrics:
                collection_name = Config.get_qdrant_collection_name(distance)

                self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="image_id",
                                    match=models.MatchValue(value=image_id),
                                )
                            ]
                        )
                    ),
                )

            return True, None

        except Exception as e:
            return False, f"Failed to delete image: {e}"

    def get_collection_stats(
        self, distance_metric: str = "cosine"
    ) -> tuple[Dict[str, Any], Optional[str]]:
        """
        Get collection statistics.

        Args:
            distance_metric: Distance metric to get stats for

        Returns:
            Tuple of (stats, error_message)
        """
        try:
            collection_name = Config.get_qdrant_collection_name(distance_metric)

            # Get collection info
            info = self.client.get_collection(collection_name)

            stats = {
                "collection_name": collection_name,
                "vectors_count": info.config.params.vectors.size,
                "distance_metric": info.config.params.vectors.distance.value,
                "status": info.status,
                "optimizer_status": info.optimizer_status,
                "config": info.config.dict(),
            }

            return stats, None

        except Exception as e:
            return {}, f"Failed to get collection stats: {e}"

    def list_collections(self) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """List all collections."""
        try:
            collections = []

            for distance in self.distance_metrics:
                collection_name = Config.get_qdrant_collection_name(distance)

                if self.client.collection_exists(collection_name=collection_name):
                    info = self.client.get_collection(collection_name)
                    collections.append(
                        {
                            "name": collection_name,
                            "vectors_count": info.config.params.vectors.size,
                            "distance_metric": info.config.params.vectors.distance.value,
                            "status": info.status,
                        }
                    )

            return collections, None

        except Exception as e:
            return [], f"Failed to list collections: {e}"

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Qdrant setup."""
        return {
            "host": self.host,
            "port": self.port,
            "collection_name": self.collection_name,
            "distance_metrics": self.distance_metrics,
            "collections_created": self.collections_created,
            "timestamp": datetime.now().isoformat(),
        }

    def test_connection(self) -> Dict[str, Any]:
        """Test connection and get system status."""
        status = {
            "component": "qdrant",
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

    def cleanup_collections(self) -> tuple[bool, Optional[str]]:
        """Clean up all collections (delete all data)."""
        try:
            for distance in self.distance_metrics:
                collection_name = Config.get_qdrant_collection_name(distance)

                if self.client.collection_exists(collection_name=collection_name):
                    self.client.delete_collection(collection_name=collection_name)
                    print(f"Deleted collection: {collection_name}")

            self.collections_created = False
            return True, "Collections cleaned up successfully"

        except Exception as e:
            return False, f"Failed to cleanup collections: {e}"
