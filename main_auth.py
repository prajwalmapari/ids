#!/usr/bin/env python3
"""
Enhanced Face Recognition System with Authorization Support
Integrates with Azure authorized persons database for security applications
"""

import cv2
import insightface
from insightface.app import FaceAnalysis
import os
import argparse
import numpy as np
from database import FaceDatabase, draw_faces_with_recognition
from azure_integration import AuthorizedPersonsManager

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

def draw_faces_with_authorization(img: np.ndarray, faces: list, recognition_results: list, 
                                 auth_manager: AuthorizedPersonsManager) -> np.ndarray:
    """Enhanced drawing function with authorization information"""
    result_img = img.copy()
    
    for i, (face, (name, confidence)) in enumerate(zip(faces, recognition_results)):
        box = face.bbox.astype(int)
        
        # Get authorization info
        auth_info = auth_manager.get_person_info(name) if name != "Unknown" else None
        
        # Choose color based on authorization status
        if name == "Unknown":
            color = (0, 0, 255)  # Red for unknown/unauthorized
            status_text = "UNAUTHORIZED"
            access_level = "NONE"
        else:
            access_level = auth_info.get('access_level', 'UNKNOWN') if auth_info else 'UNKNOWN'
            status = auth_info.get('status', 'unknown') if auth_info else 'unknown'
            
            if status.lower() == 'active' and access_level in ['HIGH', 'MEDIUM']:
                color = (0, 255, 0)  # Green for authorized
                status_text = "AUTHORIZED"
            elif status.lower() == 'active' and access_level == 'LOW':
                color = (0, 255, 255)  # Yellow for limited access
                status_text = "LIMITED ACCESS"
            else:
                color = (0, 0, 255)  # Red for inactive/unauthorized
                status_text = "UNAUTHORIZED"
        
        # Draw bounding box
        cv2.rectangle(result_img, (box[0], box[1]), (box[2], box[3]), color, 3)
        
        # Prepare labels
        main_label = f"{name} ({confidence:.2f})"
        status_label = f"{status_text} - {access_level}"
        dept_label = ""
        
        if auth_info:
            emp_id = auth_info.get('id', 'N/A')
            dept = auth_info.get('department', 'N/A')
            dept_label = f"ID: {emp_id} | Dept: {dept}"
        
        # Calculate label positions
        label_y = box[1] - 15
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Draw main label background and text
        main_size = cv2.getTextSize(main_label, font, font_scale, thickness)[0]
        cv2.rectangle(result_img, (box[0], label_y - main_size[1] - 5), 
                     (box[0] + main_size[0], label_y), color, -1)
        cv2.putText(result_img, main_label, (box[0], label_y - 5),
                   font, font_scale, (255, 255, 255), thickness)
        
        # Draw status label
        label_y -= (main_size[1] + 10)
        status_size = cv2.getTextSize(status_label, font, font_scale - 0.1, thickness)[0]
        cv2.rectangle(result_img, (box[0], label_y - status_size[1] - 5),
                     (box[0] + status_size[0], label_y), color, -1)
        cv2.putText(result_img, status_label, (box[0], label_y - 5),
                   font, font_scale - 0.1, (255, 255, 255), thickness)
        
        # Draw department info if available
        if dept_label and auth_info:
            label_y -= (status_size[1] + 10)
            dept_size = cv2.getTextSize(dept_label, font, font_scale - 0.2, 1)[0]
            cv2.rectangle(result_img, (box[0], label_y - dept_size[1] - 5),
                         (box[0] + dept_size[0], label_y), (100, 100, 100), -1)
            cv2.putText(result_img, dept_label, (box[0], label_y - 5),
                       font, font_scale - 0.2, (255, 255, 255), 1)
        
        # Draw access level indicator in corner
        access_colors = {
            'HIGH': (0, 255, 0),
            'MEDIUM': (0, 255, 255), 
            'LOW': (0, 165, 255),
            'NONE': (0, 0, 255),
            'UNKNOWN': (128, 128, 128)
        }
        access_color = access_colors.get(access_level, (128, 128, 128))
        cv2.circle(result_img, (box[2] - 15, box[1] + 15), 8, access_color, -1)
        cv2.circle(result_img, (box[2] - 15, box[1] + 15), 8, (255, 255, 255), 2)
    
    return result_img

def process_image_with_authorization(app, face_db, auth_manager, image_path, 
                                   output_path=None, show_image=True):
    """Process image with authorization checking"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    print(f"Processing image with authorization: {image_path}")
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
    
    # Print detailed authorization results
    print(f"\n=== Authorization Results ===")
    authorized_count = 0
    unauthorized_count = 0
    
    for i, ((name, confidence), face) in enumerate(zip(recognition_results, faces)):
        print(f"\nFace {i+1}:")
        print(f"  Recognition: {name} (confidence: {confidence:.3f})")
        
        if name != "Unknown":
            auth_info = auth_manager.get_person_info(name)
            if auth_info:
                print(f"  Employee ID: {auth_info.get('id', 'N/A')}")
                print(f"  Department: {auth_info.get('department', 'N/A')}")
                print(f"  Access Level: {auth_info.get('access_level', 'N/A')}")
                print(f"  Status: {auth_info.get('status', 'N/A')}")
                
                if auth_info.get('status', '').lower() == 'active':
                    print(f"  Authorization: ✅ AUTHORIZED")
                    authorized_count += 1
                else:
                    print(f"  Authorization: ❌ INACTIVE")
                    unauthorized_count += 1
            else:
                print(f"  Authorization: ❌ NOT IN SYSTEM")
                unauthorized_count += 1
        else:
            print(f"  Authorization: ❌ UNKNOWN PERSON")
            unauthorized_count += 1
        
        # Additional face attributes
        if hasattr(face, 'age'):
            print(f"  Age: {face.age:.1f}")
        if hasattr(face, 'gender'):
            gender = 'Male' if face.gender == 1 else 'Female'
            print(f"  Gender: {gender}")
    
    # Draw results with authorization info
    result_img = draw_faces_with_authorization(img, faces, recognition_results, auth_manager)
    
    # Add summary information
    summary_text = f"Faces: {len(faces)} | Authorized: {authorized_count} | Unauthorized: {unauthorized_count}"
    cv2.putText(result_img, summary_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(result_img, summary_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Add timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(result_img, f"Scan Time: {timestamp}", (10, result_img.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(result_img, f"Scan Time: {timestamp}", (10, result_img.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Save output if specified
    if output_path:
        cv2.imwrite(output_path, result_img)
        print(f"Authorization scan result saved to: {output_path}")
    
    # Display image
    if show_image:
        cv2.imshow("Authorization Face Recognition System", result_img)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print(f"\n=== Authorization Summary ===")
    print(f"✅ Authorized Personnel: {authorized_count}")
    print(f"❌ Unauthorized/Unknown: {unauthorized_count}")
    print(f"🔍 Total Faces Scanned: {len(faces)}")
    
    return result_img

def main():
    parser = argparse.ArgumentParser(description='Authorization Face Recognition System')
    parser.add_argument('--mode', choices=['test-azure', 'image', 'video', 'webcam'], 
                       default='test-azure', help='Processing mode')
    parser.add_argument('--input', type=str, help='Input image or video path')
    parser.add_argument('--output', type=str, help='Output path for results')
    parser.add_argument('--threshold', type=float, 
                       default=float(os.getenv('RECOGNITION_THRESHOLD', '0.6')),
                       help='Recognition threshold (0.0-1.0)')
    parser.add_argument('--auth-headers', type=str,
                       help='JSON string of authentication headers for Azure')
    
    args = parser.parse_args()
    
    # Initialize systems
    app = initialize_face_analysis()
    face_db = FaceDatabase("authorization_system_db.pkl")
    face_db.set_recognition_threshold(args.threshold)
    auth_manager = AuthorizedPersonsManager()
    
    if args.mode == 'test-azure':
        print("=== Testing Azure Authorized Persons Integration ===\n")
        
        # Try to load authorized persons
        auth_headers = None
        if args.auth_headers:
            import json
            try:
                auth_headers = json.loads(args.auth_headers)
            except json.JSONDecodeError:
                print("Warning: Invalid auth headers JSON, proceeding without authentication")
        
        # Load authorized persons (with fallback to sample data)
        if not auth_manager.load_from_azure(auth_headers):
            if not auth_manager.load_cache():
                print("Loading sample authorized persons for testing...")
                auth_manager.load_sample_data()
        
        # Integrate with face database
        auth_manager.integrate_with_face_database(face_db)
        
        # Test with available sample images
        sample_images = [
            '/home/ubuntu24/ids/group.png',
            '/home/ubuntu24/ids/group.png'
        ]
        
        for img_path in sample_images:
            if os.path.exists(img_path):
                print(f"\n=== Testing Authorization System ===")
                process_image_with_authorization(app, face_db, auth_manager, img_path)
                break
        else:
            print("No sample images found for testing")
            print("Add a test image and run: python main_auth.py --mode image --input your_image.jpg")
    
    elif args.mode == 'image':
        if not args.input:
            print("Please specify input image path")
            return
        
        # Load authorized persons
        if not auth_manager.load_cache():
            print("No authorized persons cache found. Loading sample data...")
            auth_manager.load_sample_data()
        
        auth_manager.integrate_with_face_database(face_db)
        process_image_with_authorization(app, face_db, auth_manager, args.input, args.output)
    
    print("\n=== Authorization System Ready ===")
    print("System Features:")
    print("✅ Face detection and recognition")
    print("✅ Authorization status checking")
    print("✅ Access level validation")
    print("✅ Employee information display")
    print("✅ Security audit logging")

if __name__ == "__main__":
    main()