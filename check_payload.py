#!/usr/bin/env python3
"""
Check the actual payload structure to understand how tags are stored.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qdrant_manager import QdrantManager
from config import Config

def check_payload_structure():
    """Check the actual payload structure of stored points."""
    print("=== Checking Payload Structure ===")
    
    try:
        qdrant_manager = QdrantManager()
        
        # Get first few points from the collection to see actual structure
        collection_name = Config.get_qdrant_collection_name(Config.get_distance_metrics()[0])
        
        try:
            # Use scroll to get actual points
            scroll_result = qdrant_manager.client.scroll(
                collection_name=collection_name,
                limit=3,
                with_payload=True
            )
            
            points = scroll_result[0]  # points are in first element of tuple
            
            if points:
                print(f"Found {len(points)} points in {collection_name}:")
                for i, point in enumerate(points):
                    print(f"\n--- Point {i+1} ---")
                    print(f"ID: {point.id}")
                    print(f"Payload keys: {list(point.payload.keys())}")
                    
                    # Check specific fields
                    if 'ai_tags' in point.payload:
                        tags = point.payload['ai_tags']
                        print(f"ai_tags: {tags}")
                        print(f"ai_tags type: {type(tags)}")
                        if isinstance(tags, list):
                            print(f"First few tags: {tags[:3]}")
                    
                    if 'ai_description' in point.payload:
                        desc = point.payload['ai_description']
                        print(f"ai_description (first 100 chars): {desc[:100]}...")
                    
                    if 'file_name' in point.payload:
                        print(f"file_name: {point.payload['file_name']}")
            else:
                print("No points found in collection")
                
        except Exception as e:
            print(f"Error getting points: {e}")
        
        return True
        
    except Exception as e:
        print(f"Debug failed: {e}")
        return False

if __name__ == "__main__":
    success = check_payload_structure()
    if success:
        print("\nPayload structure check completed.")
    else:
        print("\nPayload structure check failed.")