#!/usr/bin/env python3
"""
Test script to verify search functionality with actual data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_processor import ImageProcessor
from qdrant_manager import QdrantManager

def test_search_with_data():
    """Test search functionality with actual image data."""
    print("=== Testing Search Functionality with Data ===")
    
    try:
        # Initialize components
        processor = ImageProcessor()
        qdrant_manager = QdrantManager()
        
        print("Components initialized successfully")
        
        # Test 1: Process existing images
        print("\n--- Processing Images ---")
        project_images_dir = "project_images"
        
        if os.path.exists(project_images_dir):
            # Process images in the directory
            results, error = processor.process_bulk_images(project_images_dir, max_images=5)
            
            if error:
                print(f"Error processing images: {error}")
                return False
            else:
                print(f"Successfully processed {results.get('processed', 0)} images")
                print(f"Failed: {results.get('failed', 0)} images")
        else:
            print("No project_images directory found")
            return False
        
        # Test 2: Search for "Schnee" (snow)
        print("\n--- Searching for 'Schnee' ---")
        try:
            results, error = processor.search_images(
                text_query="Schnee",
                limit=5
            )
            
            if error:
                print(f"Search error: {error}")
                return False
            else:
                print(f"Search completed - Found {results.get('total_found', 0)} results")
                for i, result in enumerate(results.get('results', [])[:3]):  # Show first 3 results
                    payload = result.get('payload', {})
                    print(f"Result {i+1}: {payload.get('file_name', 'Unknown')}")
        except Exception as e:
            print(f"Search failed: {e}")
            return False
        
        # Test 3: Search with tags (using tags from newly processed images)
        print("\n--- Searching with Tags ---")
        try:
            results, error = processor.search_images(
                tags=["flussdiagramm", "datenbank"],
                limit=5
            )
            
            if error:
                print(f"Tag search error: {error}")
                return False
            else:
                print(f"Tag search completed - Found {results.get('total_found', 0)} results")
                for i, result in enumerate(results.get('results', [])[:3]):  # Show first 3 results
                    payload = result.get('payload', {})
                    print(f"Result {i+1}: {payload.get('file_name', 'Unknown')}")
        except Exception as e:
            print(f"Tag search failed: {e}")
            return False
        
        # Test 4: Combined search
        print("\n--- Combined Search ---")
        try:
            results, error = processor.search_images(
                text_query="datenbank",
                tags=["qdrant"],
                limit=5
            )
            
            if error:
                print(f"Combined search error: {error}")
                return False
            else:
                print(f"Combined search completed - Found {results.get('total_found', 0)} results")
                for i, result in enumerate(results.get('results', [])[:3]):  # Show first 3 results
                    payload = result.get('payload', {})
                    print(f"Result {i+1}: {payload.get('file_name', 'Unknown')}")
        except Exception as e:
            print(f"Combined search failed: {e}")
            return False
        
        print("\n=== All Data Tests Completed Successfully ===")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_search_with_data()
    if success:
        print("\n🎉 All tests with data passed! The search functionality is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")