#!/usr/bin/env python3
"""
Test script to verify the search_similar_images method fix.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_processor import ImageProcessor

def test_method_exists():
    """Test that the search_images method exists and can be called."""
    print("Testing ImageProcessor method availability...")
    
    try:
        processor = ImageProcessor()
        
        # Check if search_images method exists
        if hasattr(processor, 'search_images'):
            print("✓ search_images method exists")
            
            # Check method signature
            import inspect
            sig = inspect.signature(processor.search_images)
            print(f"✓ Method signature: {sig}")
            
            # Test with minimal parameters
            try:
                results, error = processor.search_images(limit=5)
                if error:
                    print(f"⚠ Search returned error (expected for empty database): {error}")
                else:
                    print(f"✓ Search completed successfully: {results}")
                    
            except Exception as e:
                print(f"✗ Search method call failed: {e}")
                
        else:
            print("✗ search_images method does not exist")
            
        # Check if old method name exists (should not)
        if hasattr(processor, 'search_similar_images'):
            print("⚠ search_similar_images method still exists (unexpected)")
        else:
            print("✓ search_similar_images method does not exist (as expected)")
            
    except Exception as e:
        print(f"✗ Failed to initialize ImageProcessor: {e}")

if __name__ == "__main__":
    test_method_exists()