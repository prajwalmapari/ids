import cv2
import insightface
from insightface.app import FaceAnalysis
import os
import argparse
from database import FaceDatabase, draw_faces_with_recognition

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def initialize_face_analysis():
    """Initialize the face analysis model"""
    print("Initializing face analysis model...")
    
    # Get configuration from environment variables
    model_name = os.getenv('INSIGHTFACE_MODEL_NAME', 'buffalo_l')
    providers = [os.getenv('INSIGHTFACE_PROVIDERS', 'CPUExecutionProvider')]
    det_width = int(os.getenv('DETECTION_SIZE_WIDTH', '640'))
    det_height = int(os.getenv('DETECTION_SIZE_HEIGHT', '640'))
    
    app = FaceAnalysis(name=model_name, providers=providers)
    app.prepare(ctx_id=0, det_size=(det_width, det_height))
    print("Face analysis model initialized successfully")
    return app

def process_image(app, face_db, image_path, output_path=None, show_image=True):
    """Process a single image for face detection and recognition"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    print(f"Processing image: {image_path}")
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Could not load image: {image_path}")
        return None
    
    # Detect faces
    faces = app.get(img)
    print(f"Detected {len(faces)} faces")
    
    if len(faces) == 0:
        print("No faces detected in the image")
        if show_image:
            cv2.imshow("No Faces Detected", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return img
    
    # Recognize faces
    recognition_results = face_db.recognize_faces(faces)
    
    # Print recognition results
    for i, ((name, confidence), face) in enumerate(zip(recognition_results, faces)):
        print(f"Face {i+1}: {name} (confidence: {confidence:.3f})")
        if hasattr(face, 'age'):
            print(f"  Age: {face.age:.1f}")
        if hasattr(face, 'gender'):
            gender = 'Male' if face.gender == 1 else 'Female'
            print(f"  Gender: {gender}")
    
    # Draw results on image
    result_img = draw_faces_with_recognition(img, faces, recognition_results)
    
    # Add summary text
    summary_text = f"Detected: {len(faces)} faces | Recognized: {sum(1 for name, _ in recognition_results if name != 'Unknown')}"
    cv2.putText(result_img, summary_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(result_img, summary_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    
    # Save output if specified
    if output_path:
        cv2.imwrite(output_path, result_img)
        print(f"Result saved to: {output_path}")
    
    # Display image
    if show_image:
        cv2.imshow("Multi-Target Face Recognition", result_img)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result_img

def process_video(app, face_db, video_path, output_path=None):
    """Process video for real-time face detection and recognition"""
    if video_path == '0':
        cap = cv2.VideoCapture(0)  # Webcam
        print("Starting webcam...")
    else:
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            return
        cap = cv2.VideoCapture(video_path)
        print(f"Processing video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer if output path specified
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 3rd frame for better performance
            if frame_count % 3 == 0:
                faces = app.get(frame)
                if len(faces) > 0:
                    recognition_results = face_db.recognize_faces(faces)
                    frame = draw_faces_with_recognition(frame, faces, recognition_results)
                    
                    # Add frame info
                    info_text = f"Frame: {frame_count} | Faces: {len(faces)}"
                    cv2.putText(frame, info_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write frame to output video
            if out:
                out.write(frame)
            
            # Display frame
            cv2.imshow('Multi-Target Face Recognition - Video', frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nVideo processing interrupted by user")
    
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")

def main():
    parser = argparse.ArgumentParser(description='Multi-Target Face Detection and Recognition')
    parser.add_argument('--mode', choices=['image', 'video', 'webcam', 'register'], default='image',
                       help='Processing mode')
    parser.add_argument('--input', type=str, help='Input image or video path')
    parser.add_argument('--output', type=str, help='Output path for results')
    parser.add_argument('--register-dir', type=str, default='known_faces',
                       help='Directory containing known face images for registration')
    parser.add_argument('--threshold', type=float, 
                       default=float(os.getenv('RECOGNITION_THRESHOLD', '0.6')),
                       help='Recognition threshold (0.0-1.0)')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display images (useful for batch processing)')
    
    args = parser.parse_args()
    
    # Initialize face analysis
    app = initialize_face_analysis()
    
    # Initialize face database
    face_db = FaceDatabase()
    face_db.set_recognition_threshold(args.threshold)
    
    if args.mode == 'register':
        # Register faces from directory
        if os.path.exists(args.register_dir):
            print(f"Registering faces from: {args.register_dir}")
            face_db.register_faces_from_directory(app, args.register_dir)
            print(f"Total registered faces: {face_db.get_face_count()}")
            print(f"Registered names: {face_db.list_registered_faces()}")
        else:
            print(f"Directory not found: {args.register_dir}")
            print("Please create the directory and add face images for registration")
    
    elif args.mode == 'image':
        # Process single image
        if not args.input:
            # Try default image paths
            test_images = ['group_photo.jpg', 'test.jpg', 'sample.jpg']
            for img_path in test_images:
                if os.path.exists(img_path):
                    args.input = img_path
                    break
            
            if not args.input:
                print("No input image specified and no default images found")
                print("Usage: python main.py --mode image --input <image_path>")
                return
        
        process_image(app, face_db, args.input, args.output, not args.no_display)
    
    elif args.mode == 'video':
        # Process video file
        if not args.input:
            print("Please specify input video path")
            print("Usage: python main.py --mode video --input <video_path>")
            return
        
        process_video(app, face_db, args.input, args.output)
    
    elif args.mode == 'webcam':
        # Process webcam feed
        print("Starting webcam processing. Press 'q' to quit.")
        process_video(app, face_db, '0', args.output)
    
    print("Processing complete!")

if __name__ == "__main__":
    # Quick demo if run without arguments
    import sys
    if len(sys.argv) == 1:
        print("\n=== Multi-Target Face Detection and Recognition Demo ===")
        print("\nAvailable modes:")
        print("1. Register faces: python main.py --mode register --register-dir known_faces")
        print("2. Process image: python main.py --mode image --input image.jpg")
        print("3. Process video: python main.py --mode video --input video.mp4")
        print("4. Webcam: python main.py --mode webcam")
        print("\nRunning basic demo...\n")
        
        # Initialize and run basic demo
        app = initialize_face_analysis()
        face_db = FaceDatabase()
        
        # Check if we have any test images
        test_images = ['group_photo.jpg', 'test.jpg', 'sample.jpg']
        demo_image = None
        for img_path in test_images:
            if os.path.exists(img_path):
                demo_image = img_path
                break
        
        if demo_image:
            print(f"Found test image: {demo_image}")
            process_image(app, face_db, demo_image)
        else:
            print("No test images found. Please add an image file or use command line arguments.")
            print("\nTo get started:")
            print("1. Create a 'known_faces' directory")
            print("2. Add images of people you want to recognize (name the files with their names)")
            print("3. Run: python main.py --mode register")
            print("4. Then process images: python main.py --mode image --input your_image.jpg")
    else:
        main()
