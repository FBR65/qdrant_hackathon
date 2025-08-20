"""
Gradio interface for the Qdrant Hackathon project.
"""

import gradio as gr
import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

from config import Config
from image_processor import ImageProcessor
from utils import validate_file_path, get_image_files_from_directory

# Initialize the image processor
processor = None


def initialize_processor():
    """Initialize the image processor."""
    global processor
    if processor is None:
        processor = ImageProcessor()
    return processor


def get_system_status():
    """Get system status for display."""
    try:
        processor = initialize_processor()
        status = processor.check_system_status()

        # Format status for display
        status_text = f"System Status: {status['overall_status'].upper()}\n\n"

        for component, info in status["components"].items():
            status_text += f"**{component.upper()}:** {info.get('status', 'unknown')}\n"
            if info.get("status") == "error":
                status_text += f"  Error: {info.get('error', 'Unknown error')}\n"
            elif component == "clip" and info.get("status") == "loaded":
                status_text += (
                    f"  Model: {info['model_info'].get('model_name', 'Unknown')}\n"
                )
                status_text += (
                    f"  Device: {info['model_info'].get('device', 'Unknown')}\n"
                )
            elif component == "ollama" and info.get("status") == "connected":
                status_text += (
                    f"  Model: {info['model_info'].get('model_name', 'Unknown')}\n"
                )
            elif component == "qdrant" and info.get("status") == "connected":
                collections = info.get("collections", {})
                total_collections = len(collections)
                active_collections = sum(
                    1 for c in collections.values() if "error" not in c
                )
                status_text += (
                    f"  Collections: {active_collections}/{total_collections} active\n"
                )
            status_text += "\n"

        return status_text
    except Exception as e:
        return f"Error getting system status: {e}"


def process_single_image_interface(image_file):
    """Process a single uploaded image."""
    try:
        if image_file is None:
            return "Please upload an image", None, None, None

        # Save uploaded file temporarily
        temp_path = f"temp_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        image_file.save(temp_path)

        # Process the image
        processor = initialize_processor()
        result, error = processor.process_single_image(temp_path)

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if error:
            return f"Error: {error}", None, None, None

        # Format results for display
        status = result.get("overall_status", "unknown")
        status_text = f"**Processing Status:** {status.upper()}\n\n"

        # Add steps information
        steps = result.get("steps", {})
        for step_name, step_info in steps.items():
            status_text += f"**{step_name.replace('_', ' ').title()}:** {step_info.get('status', 'unknown')}\n"
            if step_info.get("status") == "success":
                if step_name == "gps" and step_info.get("coordinates"):
                    status_text += (
                        f"  Location: {step_info.get('location', 'Unknown')}\n"
                    )
                    status_text += f"  Coordinates: {step_info['coordinates']}\n"
                elif step_name == "ollama":
                    status_text += f"  Tags: {', '.join(result.get('tags', []))}\n"
                elif step_name == "clip":
                    status_text += f"  Vector size: {step_info.get('vector_size', 0)}\n"
            status_text += "\n"

        # Create metadata display
        metadata_text = "**Image Metadata:**\n\n"
        metadata_text += f"**Filename:** {result.get('filename', 'Unknown')}\n"
        metadata_text += f"**Tags:** {', '.join(result.get('tags', []))}\n"
        metadata_text += (
            f"**Description:** {result.get('description', 'No description')}\n"
        )

        if result.get("location"):
            metadata_text += f"**Location:** {result['location']}\n"

        if result.get("coordinates"):
            metadata_text += f"**GPS:** {result['coordinates']}\n"

        metadata_text += f"**Processed:** {result.get('processed_at', 'Unknown')}\n"

        # Create search results (empty for now, will be populated when search is performed)
        search_results = []

        return status_text, metadata_text, search_results, None

    except Exception as e:
        return f"Error processing image: {e}", None, None, None


def perform_new_search(
    search_type, query_image=None, search_text="", search_tags=""
):
    """Search for similar images with forced fresh execution."""
    try:
        # Add a unique identifier to ensure fresh results and prevent caching
        import time
        search_id = f"search_{int(time.time() * 1000)}"
        
        # Force completely fresh results by using different data structures
        processor = initialize_processor()

        # Prepare search parameters
        query_path = None
        query_text = search_text.strip() if search_text else None
        search_tags_list = (
            [tag.strip() for tag in search_tags.split(",") if tag.strip()]
            if search_tags
            else None
        )

        if search_type == "image" and query_image is not None:
            # Copy the image to a temporary location to ensure it persists
            import shutil
            temp_path = f"temp_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            try:
                shutil.copy2(query_image, temp_path)
                query_path = temp_path
            except Exception as e:
                print(f"Error copying image: {e}")
                return [], f"Error processing query image: {e}"

        # Perform search
        results, error = processor.search_images(
            query_image_path=query_path,
            text_query=query_text,
            tags=search_tags_list,
            limit=10,
        )

        # Clean up temp file
        if query_path and os.path.exists(query_path):
            os.remove(query_path)

        if error:
            return [], None, f"Search error: {error}"

        # Format results for gallery
        gallery_images = []
        metadata_text = "**Search Results Metadata:**\n\n"
        
        for i, result in enumerate(results.get("results", [])):
            payload = result.get("payload", {})
            image_path = payload.get("file_path")

            if image_path and os.path.exists(image_path):
                try:
                    from PIL import Image

                    img = Image.open(image_path)
                    score = result.get("score", 0)
                    distance = result.get("distance_metric", "unknown")

                    # Create label with metadata
                    tags = payload.get("ai_tags", [])
                    location = payload.get("location_name", "")
                    label_parts = [f"Score: {score:.3f} ({distance})"]
                    if tags:
                        label_parts.append(
                            f"Tags: {', '.join(tags[:3])}{'...' if len(tags) > 3 else ''}"
                        )
                    if location:
                        label_parts.append(f"Location: {location}")

                    label = " | ".join(label_parts)
                    gallery_images.append((img, label))
                    
                    # Add detailed metadata for this result
                    metadata_text += f"--- Result {i+1} ---\n"
                    metadata_text += f"**File Path:** {image_path}\n"
                    metadata_text += f"**File Name:** {payload.get('file_name', 'Unknown')}\n"
                    metadata_text += f"**AI Tags:** {', '.join(tags) if tags else 'None'}\n"
                    metadata_text += f"**Location:** {location if location else 'None'}\n"
                    metadata_text += f"**AI Description:** {payload.get('ai_description', 'No description')[:200]}{'...' if len(payload.get('ai_description', '')) > 200 else ''}\n"
                    metadata_text += f"**Score:** {score:.3f}\n\n"
                    
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    metadata_text += f"--- Result {i+1} ---\nError loading image: {e}\n\n"
            else:
                metadata_text += f"--- Result {i+1} ---\nImage file not found\n\n"

        # Force completely fresh results by using different data structures
        if results.get("results"):
            # Create completely fresh gallery images list
            fresh_gallery = []
            for img_data in gallery_images:
                # Create new tuple to prevent reference issues
                fresh_gallery.append((img_data[0], img_data[1] + f" | ID:{search_id}"))
            
            # Create completely fresh metadata
            fresh_metadata = f"=== FRESH SEARCH RESULTS ({search_id}) ===\n{metadata_text}"
            return fresh_gallery, fresh_metadata, None
        else:
            # Return completely different structure for no results
            return [], f"=== NO RESULTS FOUND ({search_id}) ===", None

    except Exception as e:
        return [], None, f"Search interface error: {e}"


def search_images_interface(
    search_type, query_image=None, search_text="", search_tags=""
):
    """Legacy search function - not used."""
    return [], "This function is deprecated. Use perform_new_search instead.", None


def process_bulk_interface(directory_path, max_images=10):
    """Process multiple images in a directory."""
    try:
        if not directory_path.strip():
            return "Please enter a directory path", None

        directory_path = directory_path.strip()

        # Validate directory path
        if not validate_file_path(directory_path):
            return "Directory path not allowed", None

        if not os.path.exists(directory_path):
            return f"Directory not found: {directory_path}", None

        if not os.path.isdir(directory_path):
            return f"Path is not a directory: {directory_path}", None

        # Process images
        processor = initialize_processor()
        results, error = processor.process_bulk_images(directory_path, max_images)

        if error:
            return f"Error: {error}", None

        # Create summary
        total_images = results.get("total_images", 0)
        successful = results.get("processed", 0)
        failed = results.get("failed", 0)

        summary_text = f"**Bulk Processing Summary:**\n\n"
        summary_text += f"**Total Images:** {total_images}\n"
        summary_text += f"**Successful:** {successful}\n"
        summary_text += f"**Failed:** {failed}\n\n"

        # Create detailed results table
        if results.get("results"):
            data = []
            for result in results["results"]:
                filename = os.path.basename(result.get("image_path", "Unknown"))
                status = result.get("status", "unknown")
                error = result.get("error", "")

                if status == "success":
                    data = data  # This would need to be populated with actual data from successful processing
                else:
                    data.append(
                        {
                            "Filename": filename,
                            "Status": status,
                            "Error": error,
                            "Tags": "",
                            "Location": "",
                        }
                    )

            df = pd.DataFrame(data)
            return summary_text, df, None

        return summary_text, None, None

    except Exception as e:
        return f"Error in bulk processing: {e}", None, None


def set_allowed_paths_interface(paths_text):
    """Set allowed paths for file access."""
    try:
        if not paths_text.strip():
            return "Please enter at least one path"

        paths = [path.strip() for path in paths_text.split("\n") if path.strip()]

        # Validate paths exist
        valid_paths = []
        for path in paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                return f"Path does not exist: {path}"

        # Update config
        Config.set_allowed_paths(valid_paths)

        return f"Successfully set {len(valid_paths)} allowed paths:\n" + "\n".join(
            valid_paths
        )

    except Exception as e:
        return f"Error setting allowed paths: {e}"


def get_current_allowed_paths():
    """Get current allowed paths."""
    try:
        paths = Config.get_allowed_paths()
        return "\n".join(paths)
    except Exception as e:
        return f"Error getting allowed paths: {e}"


# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    # Initialize state for search results
    search_results_state = gr.State([])
    search_metadata_state = gr.State("")
    # Header
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="color: #333; margin-bottom: 10px;">🔍 Qdrant Hackathon - Image Search & Processing</h1>
        <p style="color: #666; margin: 0;">Upload, process, and search images using AI and vector databases</p>
    </div>
    """)

    # System Status
    with gr.Row():
        with gr.Column(scale=3):
            system_status = gr.Textbox(
                value=get_system_status,
                label="System Status",
                lines=8,
                interactive=False,
                elem_id="system-status",
            )
        with gr.Column(scale=1):
            refresh_status = gr.Button("🔄 Refresh Status", variant="secondary")
            refresh_status.click(fn=get_system_status, outputs=system_status)

    # Tabs
    with gr.Tabs():
        # Tab 1: Image Search
        with gr.TabItem("🔍 Image Search"):
            gr.HTML("<h3>Search for similar images</h3>")

            with gr.Row():
                with gr.Column():
                    search_type = gr.Radio(
                        choices=["text", "image"], label="Search Type", value="text"
                    )

                    with gr.Row(visible=True) as text_search_row:
                        search_text = gr.Textbox(
                            label="Search Text",
                            placeholder="Enter text to search for similar images...",
                        )

                    with gr.Row(visible=False) as image_search_row:
                        query_image = gr.Image(
                            label="Upload Query Image", type="filepath"
                        )

                    search_tags = gr.Textbox(
                        label="Search Tags (comma-separated)",
                        placeholder="tag1, tag2, tag3...",
                    )

                    search_button = gr.Button("🔍 Search Images", variant="primary")

                with gr.Column():
                    search_results = gr.Gallery(
                        label="Search Results", columns=2, height=400, interactive=False,
                        elem_id="search_results"
                    )
                    search_metadata = gr.Textbox(
                        label="Search Results Metadata",
                        lines=15,
                        interactive=False,
                        max_lines=20,
                        elem_id="search_metadata"
                    )
                    search_error = gr.Textbox(
                        label="Search Error",
                        interactive=False,
                        elem_id="search_error"
                    )

            # Show/hide search type options
            search_type.change(
                fn=lambda search_type: (
                    gr.Row(visible=search_type == "text"),
                    gr.Row(visible=search_type == "image"),
                ),
                inputs=search_type,
                outputs=[text_search_row, image_search_row],
            )

            # Search function with forced fresh outputs
            search_button.click(
                fn=perform_new_search,
                inputs=[
                    search_type,
                    query_image,
                    search_text,
                    search_tags,
                ],
                outputs=[search_results, search_metadata, search_error],
                show_progress="full"
            )

        # Tab 2: Single Image Upload
        with gr.TabItem("📤 Single Image Upload"):
            gr.HTML("<h3>Upload and process a single image</h3>")

            with gr.Row():
                with gr.Column():
                    upload_image = gr.Image(label="Upload Image", type="filepath")

                    process_button = gr.Button("🔄 Process Image", variant="primary")

                with gr.Column():
                    processing_status = gr.Textbox(
                        label="Processing Status", lines=10, interactive=False
                    )

                    image_metadata = gr.Textbox(
                        label="Image Metadata", lines=8, interactive=False
                    )

                    search_results = gr.Gallery(
                        label="Similar Images", columns=2, height=300, interactive=False
                    )

                    processing_error = gr.Textbox(label="Error", interactive=False)

            # Process function
            process_button.click(
                fn=process_single_image_interface,
                inputs=upload_image,
                outputs=[
                    processing_status,
                    image_metadata,
                    search_results,
                    processing_error,
                ],
            )

        # Tab 3: Bulk Processing
        with gr.TabItem("📦 Bulk Processing"):
            gr.HTML("<h3>Process multiple images from a directory</h3>")

            with gr.Row():
                with gr.Column():
                    directory_path = gr.Textbox(
                        label="Directory Path",
                        placeholder="Enter path to directory containing images...",
                    )

                    max_images = gr.Slider(
                        label="Max Images", minimum=1, maximum=100, value=10, step=1
                    )

                    bulk_process_button = gr.Button(
                        "🔄 Process Directory", variant="primary"
                    )

                with gr.Column():
                    bulk_summary = gr.Textbox(
                        label="Processing Summary", lines=6, interactive=False
                    )

                    bulk_results = gr.DataFrame(
                        label="Detailed Results", interactive=False
                    )

                    bulk_error = gr.Textbox(label="Error", interactive=False)

            # Bulk process function
            bulk_process_button.click(
                fn=process_bulk_interface,
                inputs=[directory_path, max_images],
                outputs=[bulk_summary, bulk_results, bulk_error],
            )

        # Tab 4: Configuration
        with gr.TabItem("⚙️ Configuration"):
            gr.HTML("<h3>System Configuration</h3>")

            with gr.Row():
                with gr.Column():
                    gr.HTML("<h4>Allowed Paths</h4>")
                    current_paths = gr.Textbox(
                        label="Current Allowed Paths",
                        value=get_current_allowed_paths,
                        lines=5,
                        interactive=False,
                    )

                    new_paths = gr.Textbox(
                        label="Set New Allowed Paths (one per line)",
                        placeholder="/path/to/images\n/another/path",
                        lines=5,
                    )

                    set_paths_button = gr.Button("📁 Set Allowed Paths")

                    paths_status = gr.Textbox(label="Paths Status", interactive=False)

                with gr.Column():
                    gr.HTML("<h4>System Information</h4>")

                    # Display configuration info
                    config_info = gr.Textbox(
                        label="Configuration",
                        value=f"""
Qdrant Host: {Config.QDRANT_HOST}
Qdrant Port: {Config.QDRANT_PORT}
Used Model: {Config.OLLAMA_MODEL}
CLIP Model: {Config.CLIP_MODEL_NAME}
Server Port: {Config.SERVER_PORT}
                        """,
                        lines=10,
                        interactive=False,
                    )

                    # Supported formats
                    formats_info = gr.Textbox(
                        label="Supported Image Formats",
                        value=", ".join(Config.SUPPORTED_EXTENSIONS),
                        interactive=False,
                    )

            # Set paths function
            set_paths_button.click(
                fn=set_allowed_paths_interface, inputs=new_paths, outputs=paths_status
            )

# Launch the interface
if __name__ == "__main__":
    demo.launch(
        server_name=Config.SERVER_NAME,
        server_port=Config.SERVER_PORT,
        share=Config.SHARE,
    )
