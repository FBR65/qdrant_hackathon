#!/usr/bin/env python3
"""
Debug script to check what's actually in Qdrant database.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qdrant_manager import QdrantManager
from image_processor import ImageProcessor

def debug_qdrant_contents():
    """Debug what's actually stored in Qdrant."""
    print("=== Debugging Qdrant Database Contents ===")
    
    try:
        qdrant_manager = QdrantManager()
        processor = ImageProcessor()
        
        # Check all collections
        collections, error = qdrant_manager.list_collections()
        if error:
            print(f"Error listing collections: {error}")
            return False
        
        print(f"Found {len(collections)} collections:")
        for collection in collections:
            print(f"  - {collection['name']}: {collection['vectors_count']} vectors")
        
        # Try to retrieve a few points from each collection
        for collection in collections:
            print(f"\n--- Checking collection: {collection['name']} ---")
            
            try:
                # Get first few points
                result = qdrant_manager.client.retrieve(
                    collection_name=collection['name'],
                    ids=["0", "1", "2"],  # Try some common IDs
                    with_payload=True
                )
                
                if result:
                    print(f"Found {len(result)} points:")
                    for i, point in enumerate(result):
                        print(f"  Point {i+1}:")
                        print(f"    ID: {point.id}")
                        print(f"    Payload keys: {list(point.payload.keys()) if point.payload else 'None'}")
                        if point.payload:
                            print(f"    Filename: {point.payload.get('file_name', 'Unknown')}")
                            print(f"    Description: {point.payload.get('ai_description', 'No description')[:100]}...")
                            print(f"    Tags: {point.payload.get('ai_tags', [])}")
                else:
                    print("No points found with IDs 0, 1, 2")
                    
                    # Try to get any point
                    try:
                        # Use scroll to get first point
                        scroll_result = qdrant_manager.client.scroll(
                            collection_name=collection['name'],
                            limit=1,
                            with_payload=True
                        )
                        
                        if scroll_result[0]:  # points are in first element of tuple
                            point = scroll_result[0][0]
                            print(f"Found first point via scroll:")
                            print(f"  ID: {point.id}")
                            print(f"  Payload keys: {list(point.payload.keys()) if point.payload else 'None'}")
                            if point.payload:
                                print(f"  Filename: {point.payload.get('file_name', 'Unknown')}")
                                print(f"  Description: {point.payload.get('ai_description', 'No description')[:100]}...")
                                print(f"  Tags: {point.payload.get('ai_tags', [])}")
                        else:
                            print("No points found in collection")
                    except Exception as scroll_error:
                        print(f"Error scrolling collection: {scroll_error}")
                        
            except Exception as e:
                print(f"Error checking collection {collection['name']}: {e}")
        
        # Test search with different approaches
        print(f"\n--- Testing Different Search Approaches ---")
        
        # Test 1: Search with exact tag match
        try:
            results, error = qdrant_manager.search_metadata(
                tags=["benutzer"],  # This is one of the tags from the first image
                limit=5
            )
            print(f"Tag search for 'benutzer': {len(results)} results, error: {error}")
        except Exception as e:
            print(f"Tag search failed: {e}")
        
        # Test 2: Search with text from description
        try:
            results, error = qdrant_manager.search_metadata(
                text_query="Bildverarbeitung",  # This is in the first image description
                limit=5
            )
            print(f"Text search for 'Bildverarbeitung': {len(results)} results, error: {error}")
        except Exception as e:
            print(f"Text search failed: {e}")
        
        # Test 3: Search with filename
        try:
            results, error = qdrant_manager.search_metadata(
                text_query="mermaid",  # This is in the filename
                limit=5
            )
            print(f"Text search for 'mermaid': {len(results)} results, error: {error}")
        except Exception as e:
            print(f"Filename search failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"Debug failed: {e}")
        return False

if __name__ == "__main__":
    success = debug_qdrant_contents()
    if success:
        print("\nDebug completed. Check the output above to see what's in the database.")
    else:
        print("\nDebug failed.")