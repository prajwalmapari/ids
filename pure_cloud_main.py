#!/usr/bin/env python3
"""
Pure Cloud Face Recognition System
Uses InsightFace for accurate processing but stores everything in Azure cloud
No local storage or dependencies beyond the processing engine
"""

import cv2
import numpy as np
import argparse
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import json
import tempfile
import base64

# InsightFace for accurate face recognition
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    raise ImportError("InsightFace is required for this cloud system. Install with: pip install insightface")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Azure Storage imports
try:
    from azure.storage.blob import BlobServiceClient, BlobClient
    from azure.core.exceptions import ResourceNotFoundError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    raise ImportError("Azure Storage SDK is required. Install with: pip install azure-storage-blob")

class PureCloudFaceProcessor:
    """Pure cloud-based face processor using InsightFace with Azure storage"""
    
    def __init__(self):
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace is required for accurate face recognition")
        
        if not AZURE_AVAILABLE:
            raise ImportError("Azure Storage SDK is required for cloud processing")
        
        # Azure configuration
        self.connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        self.container_name = os.getenv('AZURE_CONTAINER_NAME', 'sr001')
        self.embeddings_blob = os.getenv('AZURE_EMBEDDINGS_BLOB', 'authorised/authorised person/authorized_persons.json')
        self.log_blob = os.getenv('AZURE_LOG_BLOB', 'logs/face.log')
        self.images_folder = os.getenv('AZURE_IMAGES_FOLDER', 'images/unauthorized/')
        
        if not self.connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found in environment variables")
        
        # Initialize Azure client
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client = self.blob_service_client.get_container_client(self.container_name)
        
        # Initialize InsightFace
        print("🔧 Initializing InsightFace model...")
        self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ InsightFace model loaded successfully")
        
        # Load authorized persons from Azure (no local storage)
        self.authorized_persons = {}
        self.recognition_threshold = float(os.getenv('RECOGNITION_THRESHOLD', '0.6'))
        
        # Load embeddings directly from Azure
        self._load_embeddings_from_azure()
    
    def _load_embeddings_from_azure(self) -> bool:
        """Load face embeddings from Azure JSON file - no local caching"""
        try:
            print(f"📥 Loading embeddings from Azure blob: {self.embeddings_blob}")
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=self.embeddings_blob
            )
            
            # Download JSON data directly to memory
            json_data = blob_client.download_blob().readall()
            data = json.loads(json_data.decode('utf-8'))
            
            # Process authorized persons
            self.authorized_persons.clear()
            if isinstance(data, dict):
                for person_id, person_data in data.items():
                    if isinstance(person_data, dict) and 'embedding' in person_data:
                        name = person_data.get('name', person_id)
                        encoding = person_data.get('embedding', [])
                        
                        if person_id and encoding:
                            self.authorized_persons[person_id] = {
                                'name': name,
                                'encoding': np.array(encoding, dtype=np.float32),
                                'metadata': {
                                    'confidence': person_data.get('confidence', 0.0),
                                    'added_time': person_data.get('added_time', ''),
                                    'image_path': person_data.get('image_path', '')
                                },
                                'employee_id': person_id,
                                'department': person_data.get('department', 'Security'),
                                'access_level': person_data.get('access_level', 'MEDIUM'),
                                'status': person_data.get('status', 'active'),
                                'gender': person_data.get('gender', 'Unknown')
                            }
            
            print(f"✅ Successfully loaded {len(self.authorized_persons)} authorized persons from Azure")
            return True
            
        except Exception as e:
            print(f"❌ Error loading embeddings from Azure: {str(e)}")
            return False
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between InsightFace embeddings"""
        try:
            # Normalize embeddings
            embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
            embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1_norm, embedding2_norm)
            return float(similarity)
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def recognize_face_from_embedding(self, face_embedding: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """Recognize face using InsightFace embedding against Azure stored embeddings"""
        best_match = None
        best_confidence = 0.0
        best_metadata = {}
        
        try:
            for person_id, person_data in self.authorized_persons.items():
                authorized_embedding = person_data['encoding']
                
                # Calculate similarity between InsightFace embeddings
                similarity = self._calculate_similarity(face_embedding, authorized_embedding)
                
                if similarity > best_confidence and similarity >= self.recognition_threshold:
                    best_confidence = similarity
                    best_match = person_data['name']
                    best_metadata = {
                        'employee_id': person_data['employee_id'],
                        'department': person_data['department'],
                        'access_level': person_data['access_level'],
                        'status': person_data['status'],
                        'person_id': person_id
                    }
            
            if best_match:
                return best_match, best_confidence, best_metadata
            else:
                return "Unknown", 0.0, {}
        
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return "Error", 0.0, {}
    
    def _upload_image_to_azure(self, image: np.ndarray, filename: str) -> str:
        """Upload unauthorized person image to Azure storage"""
        try:
            # Convert image to bytes in memory
            success, buffer = cv2.imencode('.jpg', image)
            if not success:
                raise ValueError("Failed to encode image")
            
            image_bytes = buffer.tobytes()
            
            # Upload directly to Azure
            blob_path = f"{self.images_folder}{filename}"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_path
            )
            
            blob_client.upload_blob(image_bytes, overwrite=True)
            
            # Return the Azure blob URL
            blob_url = f"https://{blob_client.account_name}.blob.core.windows.net/{self.container_name}/{blob_path}"
            print(f"📤 Uploaded unauthorized image to Azure: {filename}")
            return blob_url
            
        except Exception as e:
            print(f"❌ Error uploading image to Azure: {e}")
            return ""
    
    def _log_prediction_to_azure(self, log_entry: Dict[str, Any]) -> bool:
        """Log face recognition prediction directly to Azure"""
        try:
            # Format log entry
            timestamp = datetime.now().isoformat()
            log_line = json.dumps({
                'timestamp': timestamp,
                **log_entry
            }) + '\n'
            
            # Download existing log file from Azure
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=self.log_blob
            )
            
            try:
                existing_content = blob_client.download_blob().readall().decode('utf-8')
            except ResourceNotFoundError:
                existing_content = ""
            
            # Append new log entry and upload back to Azure
            updated_content = existing_content + log_line
            blob_client.upload_blob(updated_content.encode('utf-8'), overwrite=True)
            
            print(f"📝 Logged prediction to Azure: {self.log_blob}")
            return True
            
        except Exception as e:
            print(f"❌ Error logging to Azure: {e}")
            return False
    
    def process_image_pure_cloud(self, image: np.ndarray, source_info: str = "") -> Dict[str, Any]:
        """Process image using InsightFace with pure cloud storage"""
        print(f"🔍 Processing image with InsightFace...")
        
        # Use InsightFace to detect and extract embeddings
        faces = self.face_app.get(image)
        
        print(f"👥 Detected {len(faces)} faces with InsightFace")
        
        results = {
            'source': source_info,
            'timestamp': datetime.now().isoformat(),
            'total_faces': len(faces),
            'authorized_count': 0,
            'unauthorized_count': 0,
            'faces': []
        }
        
        for i, face in enumerate(faces):
            face_id = f"face_{i+1}"
            
            # Extract face information from InsightFace
            bbox = face.bbox.astype(int)
            x, y, x2, y2 = bbox
            w = x2 - x
            h = y2 - y
            
            # Get face embedding from InsightFace
            face_embedding = face.normed_embedding
            
            # Recognize face using Azure embeddings
            name, confidence, metadata = self.recognize_face_from_embedding(face_embedding)
            
            # Extract face region for unauthorized persons
            face_region = image[y:y2, x:x2]
            
            # Prepare face result
            face_result = {
                'face_id': face_id,
                'recognition': {
                    'name': name,
                    'confidence': confidence,
                    'authorized': name != "Unknown" and confidence >= self.recognition_threshold
                },
                'metadata': metadata,
                'bbox': [x, y, w, h],
                'age': int(face.age) if hasattr(face, 'age') else 0,
                'gender': 'Male' if (hasattr(face, 'sex') and face.sex == 1) else 'Female',
                'detection_confidence': float(face.det_score)
            }
            
            # Handle unauthorized persons - upload to Azure
            if not face_result['recognition']['authorized']:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"unauthorized_{timestamp}_{face_id}.jpg"
                
                # Upload unauthorized image directly to Azure
                image_url = self._upload_image_to_azure(face_region, filename)
                if image_url:
                    face_result['unauthorized_image_url'] = image_url
                
                results['unauthorized_count'] += 1
            else:
                results['authorized_count'] += 1
            
            results['faces'].append(face_result)
        
        # Log all predictions to Azure (no local logging)
        log_entry = {
            'source': source_info,
            'total_faces': results['total_faces'],
            'authorized_count': results['authorized_count'],
            'unauthorized_count': results['unauthorized_count'],
            'processing_method': 'InsightFace + Azure Cloud',
            'faces': [{
                'face_id': face['face_id'],
                'name': face['recognition']['name'],
                'confidence': face['recognition']['confidence'],
                'authorized': face['recognition']['authorized'],
                'age': face['age'],
                'gender': face['gender']
            } for face in results['faces']]
        }
        
        self._log_prediction_to_azure(log_entry)
        
        return results
    
    def get_authorized_persons_count(self) -> int:
        """Get count of loaded authorized persons"""
        return len(self.authorized_persons)
    
    def get_authorized_persons_list(self) -> List[str]:
        """Get list of authorized person names"""
        return [person['name'] for person in self.authorized_persons.values()]
    
    def reload_embeddings_from_azure(self) -> bool:
        """Reload embeddings from Azure (no local caching)"""
        return self._load_embeddings_from_azure()

class PureCloudFaceRecognitionSystem:
    """Main system for pure cloud-based face recognition"""
    
    def __init__(self):
        self.cloud_processor = PureCloudFaceProcessor()
        print(f"🌐 Pure Cloud Face Recognition System initialized")
        print(f"✅ Loaded {self.cloud_processor.get_authorized_persons_count()} authorized persons from Azure")
        print(f"🔧 Processing: InsightFace | Storage: Azure Cloud | Caching: None")
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image file with pure cloud approach"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image (only temporarily for processing)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image: {image_path}")
        
        print(f"📁 Processing image: {image_path}")
        
        # Process using pure cloud approach
        results = self.cloud_processor.process_image_pure_cloud(
            image, 
            f"Image: {os.path.basename(image_path)}"
        )
        
        # Clear image from memory (no local storage)
        del image
        
        return results
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process video file with pure cloud storage"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")
        
        print(f"🎥 Processing video: {video_path}")
        frame_count = 0
        total_results = {
            'source': f"Video: {os.path.basename(video_path)}",
            'total_frames_processed': 0,
            'total_faces_detected': 0,
            'authorized_total': 0,
            'unauthorized_total': 0,
            'frame_results': []
        }
        
        # Process every 30th frame
        frame_skip = 30
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % frame_skip != 0:
                continue
            
            print(f"🎬 Processing frame {frame_count}")
            
            # Process frame with pure cloud approach
            frame_results = self.cloud_processor.process_image_pure_cloud(
                frame,
                f"Video: {os.path.basename(video_path)} - Frame {frame_count}"
            )
            
            total_results['total_frames_processed'] += 1
            total_results['total_faces_detected'] += frame_results['total_faces']
            total_results['authorized_total'] += frame_results['authorized_count']
            total_results['unauthorized_total'] += frame_results['unauthorized_count']
            total_results['frame_results'].append({
                'frame': frame_count,
                'faces': frame_results['total_faces'],
                'authorized': frame_results['authorized_count'],
                'unauthorized': frame_results['unauthorized_count']
            })
            
            # Clear frame from memory
            del frame
        
        cap.release()
        return total_results
    
    def process_webcam(self) -> None:
        """Process live webcam with pure cloud storage"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Unable to access webcam")
        
        print("📷 Starting pure cloud webcam processing...")
        print("🔧 Using InsightFace + Azure Cloud Storage")
        print("Press 'q' to quit")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Process every 15th frame for real-time performance
            if frame_count % 15 == 0:
                # Process with pure cloud approach
                results = self.cloud_processor.process_image_pure_cloud(
                    frame,
                    f"Webcam - Frame {frame_count}"
                )
                
                # Draw results on frame (no local saving)
                self._draw_results_on_frame(frame, results)
            
            # Display frame
            cv2.imshow('Pure Cloud Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _draw_results_on_frame(self, frame: np.ndarray, results: Dict[str, Any]) -> None:
        """Draw recognition results on frame"""
        for face in results['faces']:
            bbox = face.get('bbox', [])
            if len(bbox) != 4:
                continue
            
            x, y, w, h = bbox
            name = face['recognition']['name']
            confidence = face['recognition']['confidence']
            authorized = face['recognition']['authorized']
            
            # Choose colors
            color = (0, 255, 0) if authorized else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            if authorized:
                label = f"✅ {name} ({confidence:.2f})"
            else:
                label = f"❌ UNAUTHORIZED ({confidence:.2f})"
            
            # Add cloud indicator
            cv2.putText(frame, "☁️ CLOUD", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    parser = argparse.ArgumentParser(description='Pure Cloud Face Recognition System')
    parser.add_argument('--mode', choices=['image', 'video', 'webcam'], required=True,
                       help='Processing mode')
    parser.add_argument('--input', type=str, help='Input file path (for image/video modes)')
    parser.add_argument('--output', type=str, help='Output file path for results (optional)')
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = PureCloudFaceRecognitionSystem()
        
        # Process based on mode
        if args.mode == 'image':
            if not args.input:
                print("❌ Error: --input required for image mode")
                return
            
            results = system.process_image(args.input)
            
            # Print comprehensive results
            print(f"\n🌐 === Pure Cloud Processing Results ===")
            print(f"📁 Source: {results['source']}")
            print(f"👥 Total faces: {results['total_faces']}")
            print(f"✅ Authorized: {results['authorized_count']}")
            print(f"❌ Unauthorized: {results['unauthorized_count']}")
            print(f"☁️ All data stored in Azure Cloud")
            
            # Show detailed results
            print(f"\n📊 Detailed Recognition Results:")
            authorized_names = []
            for i, face in enumerate(results['faces']):
                status_icon = "✅" if face['recognition']['authorized'] else "❌"
                auth_status = "AUTHORIZED" if face['recognition']['authorized'] else "UNAUTHORIZED"
                
                print(f"\n{status_icon} Face {i+1}: {auth_status}")
                print(f"   👤 Name: {face['recognition']['name']}")
                print(f"   🎯 Confidence: {face['recognition']['confidence']:.3f}")
                print(f"   📊 Detection: {face['detection_confidence']:.3f}")
                print(f"   👶 Age: {face['age']}, 👫 Gender: {face['gender']}")
                
                if face['recognition']['authorized']:
                    authorized_names.append(face['recognition']['name'])
                    if 'metadata' in face and face['metadata']:
                        metadata = face['metadata']
                        print(f"   🆔 Employee ID: {metadata.get('employee_id', 'N/A')}")
                        print(f"   🏢 Department: {metadata.get('department', 'N/A')}")
                        print(f"   🔐 Access Level: {metadata.get('access_level', 'N/A')}")
                
                if not face['recognition']['authorized'] and 'unauthorized_image_url' in face:
                    print(f"   📤 Image uploaded to: {face['unauthorized_image_url']}")
            
            if authorized_names:
                print(f"\n🎉 Authorized personnel detected: {', '.join(set(authorized_names))}")
            
            # Save results if output specified (only metadata, no local images)
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\n💾 Results metadata saved to: {args.output}")
                print(f"☁️ All images and logs stored in Azure Cloud")
        
        elif args.mode == 'video':
            if not args.input:
                print("❌ Error: --input required for video mode")
                return
            
            results = system.process_video(args.input)
            
            print(f"\n🌐 === Pure Cloud Video Processing Results ===")
            print(f"🎥 Source: {results['source']}")
            print(f"🎬 Frames processed: {results['total_frames_processed']}")
            print(f"👥 Total faces detected: {results['total_faces_detected']}")
            print(f"✅ Total authorized: {results['authorized_total']}")
            print(f"❌ Total unauthorized: {results['unauthorized_total']}")
            print(f"☁️ All data stored in Azure Cloud")
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\n💾 Results metadata saved to: {args.output}")
        
        elif args.mode == 'webcam':
            system.process_webcam()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()