"""
Main entry point for the Qdrant Hackathon application.
"""

import sys
import os
import argparse
from typing import Optional

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import demo
from config import Config


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Qdrant Hackathon - Image Search & Processing"
    )
    parser.add_argument(
        "--host",
        default=Config.SERVER_NAME,
        help=f"Server host (default: {Config.SERVER_NAME})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=Config.SERVER_PORT,
        help=f"Server port (default: {Config.SERVER_PORT})",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=Config.SHARE,
        help="Share the application publicly",
    )
    parser.add_argument(
        "--allowed-paths", nargs="+", help="Set allowed paths for file access"
    )
    parser.add_argument(
        "--ollama-url",
        default=Config.OLLAMA_BASE_URL,
        help=f"Ollama API URL (default: {Config.OLLAMA_BASE_URL})",
    )
    parser.add_argument(
        "--ollama-model",
        default=Config.OLLAMA_MODEL,
        help=f"Ollama model name (default: {Config.OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--qdrant-host",
        default=Config.QDRANT_HOST,
        help=f"Qdrant host (default: {Config.QDRANT_HOST})",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=Config.QDRANT_PORT,
        help=f"Qdrant port (default: {Config.QDRANT_PORT})",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Update configuration with command line arguments
    Config.SERVER_NAME = args.host
    Config.SERVER_PORT = args.port
    Config.SHARE = args.share

    if args.allowed_paths:
        Config.set_allowed_paths(args.allowed_paths)

    Config.OLLAMA_BASE_URL = args.ollama_url
    Config.OLLAMA_MODEL = args.ollama_model
    Config.QDRANT_HOST = args.qdrant_host
    Config.QDRANT_PORT = args.qdrant_port

    if args.debug:
        print("=== Qdrant Hackathon Configuration ===")
        print(f"Server: {Config.SERVER_NAME}:{Config.SERVER_PORT}")
        print(f"Share: {Config.SHARE}")
        print(f"Allowed Paths: {Config.get_allowed_paths()}")
        print(f"Ollama URL: {Config.OLLAMA_BASE_URL}")
        print(f"Ollama Model: {Config.OLLAMA_MODEL}")
        print(f"Qdrant: {Config.QDRANT_HOST}:{Config.QDRANT_PORT}")
        print("=====================================")

    print("Starting Qdrant Hackathon application...")
    print(f"Access the interface at: http://{Config.SERVER_NAME}:{Config.SERVER_PORT}")

    try:
        demo.launch(
            server_name=Config.SERVER_NAME,
            server_port=Config.SERVER_PORT,
            share=Config.SHARE,
            debug=args.debug,
        )
    except KeyboardInterrupt:
        print("\nShutting down application...")
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
