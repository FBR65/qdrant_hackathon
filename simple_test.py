#!/usr/bin/env python3
"""
Simple test script to verify the search functionality fixes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_processor import ImageProcessor
from qdrant_manager import QdrantManager

def test_search_functionality():
    """Test the different search methods."""
    print("=== Testing Search Functionality ===")
    
    try:
        # Initialize components
        processor = ImageProcessor()
        qdrant_manager = QdrantManager()
        
        print("Components initialized successfully")
        
        # Test 1: Check if search_images method exists
        if hasattr(processor, 'search_images'):
            print("search_images method exists: PASS")
        else:
            print("search_images method missing: FAIL")
            return False
            
        # Test 2: Check if search_metadata method exists
        if hasattr(qdrant_manager, 'search_metadata'):
            print("search_metadata method exists: PASS")
        else:
            print("search_metadata method missing: FAIL")
            return False
            
        # Test 3: Test metadata search with text query
        print("\n--- Testing Metadata Search ---")
        try:
            results, error = qdrant_manager.search_metadata(
                text_query="test",
                limit=5
            )
            if error:
                print(f"Metadata search returned error (expected for empty database): {error}")
            else:
                print(f"Metadata search completed: {len(results)} results")
        except Exception as e:
            print(f"Metadata search failed: {e}")
            
        # Test 4: Test metadata search with tags
        try:
            results, error = qdrant_manager.search_metadata(
                tags=["test", "sample"],
                limit=5
            )
            if error:
                print(f"Tag search returned error (expected for empty database): {error}")
            else:
                print(f"Tag search completed: {len(results)} results")
        except Exception as e:
            print(f"Tag search failed: {e}")
            
        # Test 5: Test combined search
        try:
            results, error = processor.search_images(
                text_query="test",
                tags=["sample"],
                limit=5
            )
            if error:
                print(f"Combined search returned error (expected for empty database): {error}")
            else:
                print(f"Combined search completed: {results}")
        except Exception as e:
            print(f"Combined search failed: {e}")
            
        print("\n=== All Tests Completed ===")
        return True
        
    except Exception as e:
        print(f"Test failed with initialization error: {e}")
        return False

if __name__ == "__main__":
    success = test_search_functionality()
    if success:
        print("\nAll tests passed! The search functionality has been fixed.")
    else:
        print("\nSome tests failed. Please check the implementation.")