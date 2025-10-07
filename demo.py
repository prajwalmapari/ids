#!/usr/bin/env python3
"""
Demo script for multi-target face detection and recognition
This script shows basic usage of the enhanced face recognition system
"""

import cv2
import os
from main import initialize_face_analysis, process_image
from database import FaceDatabase

def create_demo_setup():
    """Create a basic demo setup with sample images"""
    print("=== Multi-Target Face Detection and Recognition Demo ===\n")
    
    # Initialize the system
    app = initialize_face_analysis()
    face_db = FaceDatabase()
    
    # Check if we have any sample images from InsightFace data
    sample_image_paths = [
        'env/insightface/data/images/t1.jpg',
        'env/insightface/data/images/Tom_Hanks_54745.png'
    ]
    
    found_images = []
    for img_path in sample_image_paths:
        if os.path.exists(img_path):
            found_images.append(img_path)
    
    if found_images:
        print(f"Found {len(found_images)} sample images:")
        for img in found_images:
            print(f"  - {img}")
        
        # Process the first available image
        demo_image = found_images[0]
        print(f"\nProcessing sample image: {demo_image}")
        result = process_image(app, face_db, demo_image)
        
        if result is not None:
            print("\nDemo completed successfully!")
            print("\nTo use with your own images:")
            print("1. Add face images to the 'known_faces' directory")
            print("2. Run: python main.py --mode register")
            print("3. Process images: python main.py --mode image --input your_image.jpg")
        
    else:
        print("No sample images found.")
        print("\nTo set up the system:")
        print("1. Create 'known_faces' directory (already created)")
        print("2. Add reference images of people you want to recognize")
        print("3. Name the files with the person's name (e.g., 'john.jpg', 'mary.png')")
        print("4. Run registration: python main.py --mode register")
        print("5. Test with images: python main.py --mode image --input test_image.jpg")
        print("\nFor webcam testing: python main.py --mode webcam")

def show_system_info():
    """Display system information and capabilities"""
    print("\n=== System Capabilities ===")
    print("✓ Multi-target face detection")
    print("✓ Face recognition with confidence scores")
    print("✓ Real-time video processing")
    print("✓ Webcam support")
    print("✓ Batch processing")
    print("✓ Persistent face database")
    print("✓ Age and gender detection (when available)")
    print("✓ Facial landmarks detection")
    print("✓ Configurable recognition thresholds")
    
    print("\n=== Supported Formats ===")
    print("Images: JPG, JPEG, PNG, BMP")
    print("Videos: MP4, AVI, MOV (OpenCV supported formats)")
    print("Input: Files, directories, webcam")
    
    print("\n=== Command Examples ===")
    print("Register faces:")
    print("  python main.py --mode register --register-dir known_faces")
    print("\nProcess image:")
    print("  python main.py --mode image --input group_photo.jpg")
    print("\nProcess video:")
    print("  python main.py --mode video --input video.mp4")
    print("\nWebcam (real-time):")
    print("  python main.py --mode webcam")
    print("\nAdjust threshold:")
    print("  python main.py --mode image --input photo.jpg --threshold 0.7")

if __name__ == "__main__":
    show_system_info()
    create_demo_setup()