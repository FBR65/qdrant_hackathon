#!/usr/bin/env python3
"""
Test script for GPS/EXIF processing functionality.
"""

import sys
import os

sys.path.append(".")

from utils import get_gps_coordinates, get_location_from_coordinates


def test_gps_processing():
    """Test the GPS/EXIF processing functionality."""
    print("=== GPS/EXIF Processing Test ===")

    # Test mit dem vorhandenen PNG-Bild (sollte keine GPS-Daten haben)
    test_image = "project_images/mermaid_diagram_1755449238483.png"
    if os.path.exists(test_image):
        print(f"Testing GPS extraction from: {test_image}")
        gps_coords = get_gps_coordinates(test_image)
        print(f"GPS coordinates: {gps_coords}")

        if gps_coords:
            lat, lon = gps_coords
            location = get_location_from_coordinates(lat, lon)
            print(f"Location from coordinates: {location}")
        else:
            print("No GPS coordinates found (expected for PNG)")
    else:
        print(f"Test image not found: {test_image}")

    # Test mit einem Beispiel-Bild mit GPS-Daten (wenn vorhanden)
    print("\nLooking for images with GPS data...")
    image_dir = "project_images"
    if os.path.exists(image_dir):
        for filename in os.listdir(image_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".tiff")):
                image_path = os.path.join(image_dir, filename)
                gps_coords = get_gps_coordinates(image_path)
                if gps_coords:
                    print(f"Found GPS data in: {filename}")
                    lat, lon = gps_coords
                    location = get_location_from_coordinates(lat, lon)
                    print(f"  Coordinates: {lat}, {lon}")
                    print(f"  Location: {location}")
                    break
        else:
            print("No images with GPS data found in the project_images directory")
    else:
        print("Project images directory not found")

    # Test mit Mock-GPS-Koordinaten
    print("\n=== Mock GPS Test ===")
    mock_lat = 52.5200  # Berlin coordinates
    mock_lon = 13.4050
    location = get_location_from_coordinates(mock_lat, mock_lon)
    print(f"Mock coordinates: {mock_lat}, {mock_lon}")
    print(f"Mock location: {location}")


if __name__ == "__main__":
    test_gps_processing()
