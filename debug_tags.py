#!/usr/bin/env python3
"""
Debug script to check tag storage and search.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qdrant_manager import QdrantManager

def debug_tag_storage():
    """Debug how tags are stored and searched."""
    print("=== Debugging Tag Storage and Search ===")
    
    try:
        qdrant_manager = QdrantManager()
        
        # Test different tag search approaches
        test_tags = [
            "flussdiagramm",  # From first image
            "datenbankdiagramm",  # From second image
            "benutzer",  # From first image
            "qdrant",  # From second image
        ]
        
        print("Testing individual tag searches:")
        for tag in test_tags:
            try:
                results, error = qdrant_manager.search_metadata(
                    tags=[tag],
                    limit=5
                )
                print(f"  Tag '{tag}': {len(results)} results, error: {error}")
                
                if results:
                    for result in results[:1]:
                        payload = result.get('payload', {})
                        print(f"    Found: {payload.get('file_name', 'Unknown')}")
            except Exception as e:
                print(f"  Tag '{tag}': Exception - {e}")
        
        # Test with multiple tags
        print(f"\nTesting multiple tags: ['flussdiagramm', 'datenbankdiagramm']")
        try:
            results, error = qdrant_manager.search_metadata(
                tags=["flussdiagramm", "datenbankdiagramm"],
                limit=5
            )
            print(f"  Multiple tags: {len(results)} results, error: {error}")
        except Exception as e:
            print(f"  Multiple tags: Exception - {e}")
        
        # Test text search with known content
        print(f"\nTesting text search with known content:")
        search_terms = [
            "Flussdiagramm",  # Should match first image description
            "Datenbankdiagramm",  # Should match second image description
            "Qdrant",  # Should match second image
            "Benutzer",  # Should match first image
        ]
        
        for term in search_terms:
            try:
                results, error = qdrant_manager.search_metadata(
                    text_query=term,
                    limit=5
                )
                print(f"  Text '{term}': {len(results)} results, error: {error}")
                
                if results:
                    for result in results[:1]:
                        payload = result.get('payload', {})
                        print(f"    Found: {payload.get('file_name', 'Unknown')}")
            except Exception as e:
                print(f"  Text '{term}': Exception - {e}")
        
        return True
        
    except Exception as e:
        print(f"Debug failed: {e}")
        return False

if __name__ == "__main__":
    success = debug_tag_storage()
    if success:
        print("\nTag debugging completed.")
    else:
        print("\nTag debugging failed.")