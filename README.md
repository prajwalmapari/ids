# Multi-Target Face Detection and Recognition System

This project implements a comprehensive face detection and recognition system capable of handling multiple faces simultaneously using InsightFace and OpenCV.

## Features

- **Multi-target face detection**: Detect multiple faces in a single image or video frame
- **Face recognition**: Recognize known individuals with confidence scores
- **Real-time processing**: Support for webcam and video file processing
- **Persistent database**: Store and manage face embeddings with metadata
- **Flexible input**: Support for images, videos, and live camera feeds
- **Batch processing**: Register multiple faces from a directory
- **Age and gender detection**: Additional face attributes when available
- **Configurable thresholds**: Adjust recognition sensitivity

## Installation

1. Make sure you have Python 3.8+ installed
2. Install required packages:
```bash
pip install insightface opencv-python numpy
```

3. The system will automatically download the face analysis model on first run.

## Quick Start

### 1. Register Known Faces

Create a `known_faces` directory and add images of people you want to recognize:

```bash
mkdir known_faces
# Add images named with person's name: john.jpg, mary.png, etc.
python main.py --mode register --register-dir known_faces
```

### 2. Process Images

```bash
# Process a single image
python main.py --mode image --input group_photo.jpg

# Save output
python main.py --mode image --input photo.jpg --output result.jpg
```

### 3. Process Videos

```bash
# Process video file
python main.py --mode video --input video.mp4

# Real-time webcam
python main.py --mode webcam
```

## Usage Examples

### Basic Commands

```bash
# Run demo
python demo.py

# Register faces from directory
python main.py --mode register --register-dir known_faces

# Process image with custom threshold
python main.py --mode image --input test.jpg --threshold 0.7

# Process video without display (batch mode)
python main.py --mode video --input video.mp4 --no-display --output processed_video.mp4
```

### Command Line Options

- `--mode`: Processing mode (image, video, webcam, register)
- `--input`: Input file path
- `--output`: Output file path
- `--register-dir`: Directory with face images for registration
- `--threshold`: Recognition threshold (0.0-1.0, default: 0.6)
- `--no-display`: Disable image display (for batch processing)

## System Architecture

### Core Components

1. **FaceDatabase**: Manages face embeddings and metadata
2. **Face Analysis**: InsightFace-based detection and feature extraction
3. **Recognition Engine**: Similarity matching with configurable thresholds
4. **Visualization**: Drawing bounding boxes, labels, and confidence scores

### Recognition Process

1. **Detection**: Locate faces in the image using InsightFace
2. **Feature Extraction**: Generate normalized embeddings for each face
3. **Matching**: Compare embeddings with registered faces using cosine similarity
4. **Classification**: Assign identity based on highest similarity above threshold
5. **Visualization**: Draw results with bounding boxes and labels

## Performance Tips

- **Video Processing**: Processes every 3rd frame for better performance
- **Batch Mode**: Use `--no-display` for faster processing
- **Threshold Tuning**: Lower values = more strict recognition
- **Image Size**: Larger detection size (det_size) = better accuracy but slower

## File Structure

```
ids/
├── main.py              # Main application with CLI interface
├── database.py          # Face database management
├── demo.py              # Demo script and examples
├── known_faces/         # Directory for reference face images
├── face_database.pkl    # Persistent face database (auto-generated)
└── env/                 # Virtual environment
```

## Troubleshooting

### Common Issues

1. **No faces detected**: Check image quality and lighting
2. **Poor recognition**: Adjust threshold or add more reference images
3. **Slow performance**: Reduce detection size or use batch mode
4. **Memory issues**: Process videos in chunks or reduce resolution

### Error Messages

- "No input image specified": Provide image path with `--input`
- "Directory not found": Create `known_faces` directory first
- "Could not load image": Check file format and path
- "No faces detected": Ensure faces are visible and well-lit

## Advanced Usage

### Custom Thresholds

```python
from database import FaceDatabase

face_db = FaceDatabase()
face_db.set_recognition_threshold(0.8)  # More strict
```

### Programmatic Usage

```python
from main import initialize_face_analysis, process_image
from database import FaceDatabase

app = initialize_face_analysis()
face_db = FaceDatabase()
result = process_image(app, face_db, "test.jpg")
```

## Contributing

Feel free to contribute improvements, bug fixes, or new features!

## License

This project uses InsightFace for face analysis. Please check their license terms for commercial usage.