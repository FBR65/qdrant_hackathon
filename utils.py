"""
Utility functions for the Qdrant Hackathon project.
"""

import os
import base64
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from PIL import Image, UnidentifiedImageError
import piexif
import piexif.helper
import exifread
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

from config import Config


def _sanitize_for_json(data: Any) -> Any:
    """Make data JSON serializable."""
    if isinstance(data, bytes):
        try:
            return data.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            return f"base64:{base64.b64encode(data).decode('ascii')}"
    elif isinstance(data, dict):
        return {k: _sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_sanitize_for_json(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(_sanitize_for_json(item) for item in data)
    elif isinstance(data, (int, float, str, bool, type(None))):
        return data
    else:
        return repr(data)


def _get_exif_with_names(exif_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Get human-readable EXIF tag names."""
    exif_with_names = {}
    for ifd_name, ifd_dict in exif_dict.items():
        if ifd_name == "thumbnail":
            exif_with_names[ifd_name] = "thumbnail_data"
            continue
        if not isinstance(ifd_dict, dict):
            continue

        exif_with_names[ifd_name] = {}
        for tag, value in ifd_dict.items():
            tag_name = exifread.TAGS.get(tag, tag)
            if ifd_name == "GPS":
                gps_tag_name = exifread.GPSTAGS.get(tag, tag)
                exif_with_names[ifd_name][gps_tag_name] = value
            else:
                exif_with_names[ifd_name][tag_name] = value
    return exif_with_names


def _convert_to_degrees(value) -> Optional[float]:
    """Convert GPS coordinates to degrees."""
    if not hasattr(value, "__len__") or len(value) < 3:
        return None

    try:
        d = (
            float(value[0].num) / float(value[0].den)
            if value[0].den != 0
            else float(value[0].num)
        )
        m = (
            float(value[1].num) / float(value[1].den)
            if value[1].den != 0
            else float(value[1].num)
        )
        s = (
            float(value[2].num) / float(value[2].den)
            if value[2].den != 0
            else float(value[2].num)
        )
        return d + (m / 60.0) + (s / 3600.0)
    except (ZeroDivisionError, AttributeError, IndexError, ValueError) as e:
        print(f"Error converting GPS rational to degrees: {e}")
        return None


def get_gps_coordinates(image_path: str) -> Optional[Tuple[float, float]]:
    """Extract GPS coordinates from image."""
    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, stop_tag="GPS GPSLongitude")

        if not tags:
            return None

        gps_latitude = tags.get("GPS GPSLatitude")
        gps_latitude_ref = tags.get("GPS GPSLatitudeRef")
        gps_longitude = tags.get("GPS GPSLongitude")
        gps_longitude_ref = tags.get("GPS GPSLongitudeRef")

        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = _convert_to_degrees(gps_latitude.values)
            lon = _convert_to_degrees(gps_longitude.values)

            if lat is None or lon is None:
                return None

            if gps_latitude_ref.values[0] == "S":
                lat = -lat
            if gps_longitude_ref.values[0] == "W":
                lon = -lon

            return lat, lon
        return None
    except Exception as e:
        print(f"Error reading GPS data: {e}")
        return None


def get_location_from_coordinates(latitude: float, longitude: float) -> Optional[str]:
    """Get location name from GPS coordinates."""
    try:
        geolocator = Nominatim(user_agent="qdrant_hackathon_v1.0")
        location = geolocator.reverse(
            (latitude, longitude), exactly_one=True, language="en", timeout=10
        )
        return (
            location.address.strip().lower() if location and location.address else None
        )
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Geocoding error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected geocoding error: {e}")
        return None


def extract_and_add_metadata(
    image_path: str, tags: List[str]
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """Extract metadata and add tags to image."""
    tags_str = ", ".join(sorted(list(set(tags))))
    extracted_metadata = {}
    write_success = False
    error_msg = None

    try:
        with Image.open(image_path) as img:
            img_format = img.format

            if img_format in ["JPEG", "TIFF"]:
                try:
                    exif_dict = piexif.load(img.info.get("exif", b""))
                    extracted_metadata = _get_exif_with_names(exif_dict)
                except Exception as e:
                    print(f"Warning: Problem loading existing EXIF: {e}")
                    exif_dict = {
                        "0th": {},
                        "Exif": {},
                        "GPS": {},
                        "1st": {},
                        "thumbnail": None,
                    }
                    extracted_metadata = {"Info": img.info.copy()} if img.info else {}

                try:
                    if "Exif" not in exif_dict:
                        exif_dict["Exif"] = {}
                    exif_dict["Exif"][piexif.ExifIFD.UserComment] = (
                        piexif.helper.UserComment.dump(tags_str, encoding="unicode")
                    )
                    exif_bytes = piexif.dump(exif_dict)

                    save_kwargs = {"exif": exif_bytes}
                    if img_format == "JPEG":
                        save_kwargs["quality"] = img.info.get("quality", 95)
                        save_kwargs["subsampling"] = img.info.get("subsampling", -1)
                        save_kwargs["progressive"] = img.info.get("progressive", False)
                        save_kwargs["icc_profile"] = img.info.get("icc_profile")

                    img.save(image_path, **save_kwargs)
                    write_success = True

                    if "Exif" not in extracted_metadata:
                        extracted_metadata["Exif"] = {}
                    extracted_metadata["Exif"]["UserComment"] = tags_str

                except Exception as write_e:
                    error_msg = f"Failed to write EXIF tags: {write_e}"

            elif img_format == "PNG":
                try:
                    existing_info = img.info or {}
                    extracted_metadata = {"Info": existing_info.copy()}

                    from PIL.PngImagePlugin import PngInfo

                    pnginfo = PngInfo()

                    for k, v in existing_info.items():
                        if isinstance(v, str) and k.lower() != "keywords":
                            pnginfo.add_text(k, v)

                    pnginfo.add_itxt("Keywords", tags_str, lang="en", tkey="Keywords")
                    img.save(image_path, pnginfo=pnginfo)
                    write_success = True

                    if "Info" not in extracted_metadata:
                        extracted_metadata["Info"] = {}
                    extracted_metadata["Info"]["Keywords"] = tags_str

                except Exception as png_e:
                    error_msg = f"Error processing PNG metadata: {png_e}"

            else:
                error_msg = f"Unsupported format ({img_format}) for metadata writing."
                try:
                    info_dict = img.info
                    if info_dict:
                        extracted_metadata = {"Info": info_dict.copy()}
                    else:
                        extracted_metadata = {}
                except Exception:
                    extracted_metadata = {}

    except UnidentifiedImageError:
        error_msg = "Cannot identify image file"
    except FileNotFoundError:
        error_msg = "Image file not found"
    except PermissionError:
        error_msg = "Permission denied"
    except Exception as e:
        error_msg = f"Unexpected error processing metadata: {e}"

    if extracted_metadata is None:
        extracted_metadata = {}

    return extracted_metadata, write_success, error_msg


def is_image_file(filepath: str) -> bool:
    """Check if file is a supported image format."""
    return filepath.lower().endswith(tuple(Config.SUPPORTED_EXTENSIONS))


def get_image_files_from_directory(directory: str) -> List[str]:
    """Get all supported image files from directory."""
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if is_image_file(file):
                image_files.append(os.path.join(root, file))
    return image_files


def validate_file_path(filepath: str) -> bool:
    """Validate if file path is allowed."""
    allowed_paths = Config.get_allowed_paths()
    abs_path = os.path.abspath(filepath)

    for allowed_path in allowed_paths:
        abs_allowed = os.path.abspath(allowed_path)
        if abs_path.startswith(abs_allowed):
            return True
    return False


def create_payload(image_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create payload for Qdrant from image data."""
    return {
        "image_id": image_data.get("image_id", ""),
        "file_path": image_data.get("file_path", ""),
        "file_name": image_data.get("file_name", ""),
        "file_size": image_data.get("file_size", 0),
        "width": image_data.get("width", 0),
        "height": image_data.get("height", 0),
        "format": image_data.get("format", ""),
        "processing_timestamp": image_data.get("processing_timestamp", ""),
        "gps_coordinates": image_data.get("gps_coordinates", []),
        "location_name": image_data.get("location_name", ""),
        "ai_tags": image_data.get("ai_tags", []),
        "ai_description": image_data.get("ai_description", ""),
        "model_used": image_data.get("model_used", ""),
        "embedding_dim": image_data.get("embedding_dim", 0),
        "processed_at": datetime.now().isoformat(),
        "source_type": "upload"
        if os.path.exists(image_data.get("file_path", ""))
        else "processing",
    }
