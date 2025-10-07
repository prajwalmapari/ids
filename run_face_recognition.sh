#!/bin/bash

# Multi-Target Face Detection and Recognition - Easy Usage Script
# This script makes it easy to run the face recognition system

PYTHON_CMD="/home/ubuntu24/ids/env/bin/python"
SCRIPT_DIR="/home/ubuntu24/ids"

cd "$SCRIPT_DIR"

echo "=== Multi-Target Face Detection and Recognition System ==="
echo

# Function to show usage
show_usage() {
    echo "Usage: $0 [mode] [options]"
    echo
    echo "Modes:"
    echo "  demo        - Run interactive demo"
    echo "  test        - Run system test"
    echo "  register    - Register faces from known_faces directory"
    echo "  image       - Process a single image"
    echo "  video       - Process a video file"
    echo "  webcam      - Use webcam for real-time detection"
    echo
    echo "Examples:"
    echo "  $0 demo"
    echo "  $0 register"
    echo "  $0 image photo.jpg"
    echo "  $0 video video.mp4"
    echo "  $0 webcam"
    echo
}

# Check if Python environment is available
if [ ! -f "$PYTHON_CMD" ]; then
    echo "Error: Python environment not found at $PYTHON_CMD"
    echo "Please check your installation."
    exit 1
fi

# Handle different modes
case "$1" in
    "demo")
        echo "Running interactive demo..."
        $PYTHON_CMD demo.py
        ;;
    "test")
        echo "Running system test..."
        $PYTHON_CMD test_system.py
        ;;
    "register")
        echo "Registering faces from known_faces directory..."
        if [ ! -d "known_faces" ]; then
            echo "Creating known_faces directory..."
            mkdir -p known_faces
            echo "Please add face images to the known_faces directory and run this command again."
            exit 1
        fi
        $PYTHON_CMD main.py --mode register --register-dir known_faces
        ;;
    "image")
        if [ -z "$2" ]; then
            echo "Error: Please specify an image file"
            echo "Usage: $0 image <image_file>"
            exit 1
        fi
        if [ ! -f "$2" ]; then
            echo "Error: Image file '$2' not found"
            exit 1
        fi
        echo "Processing image: $2"
        $PYTHON_CMD main.py --mode image --input "$2"
        ;;
    "video")
        if [ -z "$2" ]; then
            echo "Error: Please specify a video file"
            echo "Usage: $0 video <video_file>"
            exit 1
        fi
        if [ ! -f "$2" ]; then
            echo "Error: Video file '$2' not found"
            exit 1
        fi
        echo "Processing video: $2"
        echo "Press 'q' to quit video processing"
        $PYTHON_CMD main.py --mode video --input "$2"
        ;;
    "webcam")
        echo "Starting webcam face detection..."
        echo "Press 'q' to quit"
        $PYTHON_CMD main.py --mode webcam
        ;;
    "-h"|"--help"|"help"|"")
        show_usage
        ;;
    *)
        echo "Error: Unknown mode '$1'"
        echo
        show_usage
        exit 1
        ;;
esac