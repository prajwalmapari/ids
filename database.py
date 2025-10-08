import cv2
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class FaceDatabase:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.getenv('FACE_DATABASE_FILE', 'face_database.pkl')
        self.known_faces: Dict[str, np.ndarray] = {}
        self.face_metadata: Dict[str, dict] = {}
        self.recognition_threshold = float(os.getenv('RECOGNITION_THRESHOLD', '0.6'))
        self.load_database()
    
    def load_database(self):
        """Load face database from file"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data.get('faces', {})
                    self.face_metadata = data.get('metadata', {})
                print(f"Loaded {len(self.known_faces)} faces from database")
            except Exception as e:
                print(f"Error loading database: {e}")
    
    def save_database(self):
        """Save face database to file"""
        try:
            data = {
                'faces': self.known_faces,
                'metadata': self.face_metadata
            }
            with open(self.db_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved {len(self.known_faces)} faces to database")
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def register_face(self, name: str, embedding: np.ndarray, image_path: str = None):
        """Register a new face in the database"""
        self.known_faces[name] = embedding
        self.face_metadata[name] = {
            'registered_date': str(np.datetime64('now')),
            'image_path': image_path,
            'embedding_size': embedding.shape[0]
        }
        self.save_database()
        print(f"Registered face: {name}")
    
    def register_faces_from_directory(self, app, directory: str):
        """Register all faces from a directory of images"""
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist")
            return
        
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(directory, filename)
                
                try:
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Could not load image: {image_path}")
                        continue
                    
                    faces = app.get(img)
                    if len(faces) == 0:
                        print(f"No face detected in {filename}")
                        continue
                    elif len(faces) > 1:
                        print(f"Multiple faces detected in {filename}, using the largest one")
                        # Use the face with the largest bounding box
                        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                    
                    face = faces[0]
                    self.register_face(name, face.normed_embedding, image_path)
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    
    def recognize_faces(self, faces: List) -> List[Tuple[str, float]]:
        """Recognize multiple faces and return list of (name, confidence) tuples"""
        results = []
        
        for face in faces:
            emb = face.normed_embedding
            best_name = "Unknown"
            best_similarity = 0.0
            
            for name, db_emb in self.known_faces.items():
                # Calculate cosine similarity instead of L2 distance
                similarity = np.dot(emb, db_emb) / (np.linalg.norm(emb) * np.linalg.norm(db_emb))
                
                if similarity > best_similarity and similarity > self.recognition_threshold:
                    best_name = name
                    best_similarity = similarity
            
            results.append((best_name, best_similarity))
        
        return results
    
    def get_face_count(self) -> int:
        """Get number of registered faces"""
        return len(self.known_faces)
    
    def list_registered_faces(self) -> List[str]:
        """Get list of all registered face names"""
        return list(self.known_faces.keys())
    
    def remove_face(self, name: str) -> bool:
        """Remove a face from the database"""
        if name in self.known_faces:
            del self.known_faces[name]
            if name in self.face_metadata:
                del self.face_metadata[name]
            self.save_database()
            print(f"Removed face: {name}")
            return True
        return False
    
    def set_recognition_threshold(self, threshold: float):
        """Set the recognition threshold (0.0 to 1.0)"""
        self.recognition_threshold = max(0.0, min(1.0, threshold))
        print(f"Recognition threshold set to: {self.recognition_threshold}")

def draw_faces_with_recognition(img: np.ndarray, faces: List, recognition_results: List[Tuple[str, float]]) -> np.ndarray:
    """Draw bounding boxes and labels on detected faces"""
    result_img = img.copy()
    
    for i, (face, (name, confidence)) in enumerate(zip(faces, recognition_results)):
        box = face.bbox.astype(int)
        
        # Choose color based on recognition result
        if name == "Unknown":
            color = (0, 0, 255)  # Red for unknown
            label = f"Unknown ({confidence:.2f})"
        else:
            color = (0, 255, 0)  # Green for recognized
            label = f"{name} ({confidence:.2f})"
        
        # Draw bounding box
        cv2.rectangle(result_img, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(result_img, (box[0], box[1] - label_size[1] - 10), 
                     (box[0] + label_size[0], box[1]), color, -1)
        
        # Draw label text
        cv2.putText(result_img, label, (box[0], box[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw face landmarks if available
        if hasattr(face, 'kps') and face.kps is not None:
            kps = face.kps.astype(int)
            for kp in kps:
                cv2.circle(result_img, tuple(kp), 2, color, -1)
    
    return result_img
