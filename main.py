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
import multiprocessing
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache
import weakref
import gc

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

class SystemOptimizer:
    """Root-level system optimization based on hardware detection"""
    
    def __init__(self):
        self.cpu_cores = multiprocessing.cpu_count()
        self.memory_gb = self._get_available_memory()
        self.is_arm = self._detect_arm_architecture()
        self.optimal_threads = max(1, self.cpu_cores - 2)  # Leave cores for system
        self.setup_environment()
    
    def _get_available_memory(self) -> float:
        """Get available system memory in GB"""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        kb = int(line.split()[1])
                        return kb / (1024 * 1024)  # Convert to GB
            return 4.0  # Default fallback
        except:
            return 4.0
    
    def _detect_arm_architecture(self) -> bool:
        """Detect if running on ARM (Jetson/Pi) architecture"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read().lower()
                return any(arch in content for arch in ['arm', 'aarch64', 'cortex'])
        except:
            return False
    
    def setup_environment(self):
        """Configure environment based on detected hardware"""
        # Dynamic thread optimization
        os.environ['OMP_NUM_THREADS'] = str(self.optimal_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.optimal_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.optimal_threads)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(self.optimal_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.optimal_threads)
        
        # Memory optimization
        if self.memory_gb < 8:
            os.environ['OPENCV_VIDEOCAPTURE_CACHE_SIZE'] = '1'
            os.environ['OPENCV_DNN_OPENCL_ALLOW_ALL_DEVICES'] = '0'
        
        # ARM-specific optimizations
        if self.is_arm:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU on ARM
            os.environ['OPENCV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2_USE_NEW_API'] = '1'
    
    def get_optimal_detection_size(self) -> tuple:
        """Get optimal detection size based on hardware capabilities"""
        if self.memory_gb < 4:
            return (320, 320)
        elif self.memory_gb < 8:
            return (416, 416)
        else:
            return (640, 640)
    
    def get_processing_config(self) -> dict:
        """Get processing configuration based on hardware"""
        return {
            'max_faces': min(16, max(4, int(self.memory_gb * 2))),
            'frame_skip': max(1, 6 - int(self.memory_gb)),
            'thread_pool_size': self.optimal_threads,
            'batch_size': min(8, max(1, int(self.memory_gb / 2))),
            'enable_caching': self.memory_gb > 4
        }

# Initialize system optimizer
system_optimizer = SystemOptimizer()

# Azure Storage imports
try:
    from azure.storage.blob import BlobServiceClient, BlobClient
    from azure.core.exceptions import ResourceNotFoundError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    raise ImportError("Azure Storage SDK is required. Install with: pip install azure-storage-blob")

class OptimizedFaceProcessor:
    """Hardware-optimized face processor with intelligent resource management"""
    
    def __init__(self):
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace is required for accurate face recognition")
        
        if not AZURE_AVAILABLE:
            raise ImportError("Azure Storage SDK is required for cloud processing")
        
        # Get hardware-optimized configuration
        self.config = system_optimizer.get_processing_config()
        self.detection_size = system_optimizer.get_optimal_detection_size()
        
        # Azure configuration
        self.connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        self.container_name = os.getenv('AZURE_CONTAINER_NAME', 'sr001')
        self.embeddings_blob = os.getenv('AZURE_EMBEDDINGS_BLOB', 'authorised/authorised person/authorized_persons.json')
        self.log_blob = os.getenv('AZURE_LOG_BLOB', 'unauthorised_person/detection_logs/')
        self.images_folder = os.getenv('AZURE_IMAGES_FOLDER', 'unauthorised_person/Image/')
        
        if not self.connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found in environment variables")
        
        # Initialize Azure client with connection pooling
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.connection_string,
            max_single_get_size=4*1024*1024,
            max_chunk_get_size=4*1024*1024
        )
        self.container_client = self.blob_service_client.get_container_client(self.container_name)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.config['thread_pool_size'])
        
        # Embedding cache with weak references for memory efficiency
        self._embedding_cache = weakref.WeakValueDictionary() if self.config['enable_caching'] else None
        
        # Initialize InsightFace with hardware-optimized settings
        print(f"🔧 Initializing InsightFace model (Hardware: {system_optimizer.cpu_cores} cores, {system_optimizer.memory_gb:.1f}GB RAM)...")
        
        # Dynamic provider configuration based on hardware
        providers = ['CPUExecutionProvider']
        provider_options = [{
            'CPUExecutionProvider': {
                'enable_cpu_mem_arena': True,
                'arena_extend_strategy': 'kNextPowerOfTwo' if system_optimizer.memory_gb > 4 else 'kSameAsRequested',
                'cpu_mem_limit': int(system_optimizer.memory_gb * 0.4 * 1024 * 1024 * 1024),  # 40% of available RAM
                'enable_mem_pattern': True,
                'enable_mem_reuse': True,
                'intra_op_num_threads': system_optimizer.optimal_threads,
                'inter_op_num_threads': max(1, system_optimizer.optimal_threads // 2)
            }
        }]
        
        self.face_app = FaceAnalysis(providers=providers)
        self.face_app.prepare(ctx_id=-1, det_size=self.detection_size)
        
        print(f"✅ InsightFace initialized - Detection: {self.detection_size}, Threads: {system_optimizer.optimal_threads}")
        print(f"🎯 Config: Max faces: {self.config['max_faces']}, Batch size: {self.config['batch_size']}")
        
        # Load authorized persons with caching optimization
        self.authorized_persons = {}
        self.recognition_threshold = float(os.getenv('RECOGNITION_THRESHOLD', '0.6'))
        
        # Performance monitoring
        self._processing_stats = {
            'total_processed': 0,
            'avg_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        print(f"🚀 Hardware optimizations enabled")
        print(f"👥 Max faces per frame: {self.config['max_faces']}")
        print(f"📹 Frame skip ratio: {self.config['frame_skip']}")
        print(f"🧵 Thread pool size: {self.config['thread_pool_size']}")
        print(f"💾 Caching enabled: {self.config['enable_caching']}")
        
        # Load embeddings with optimization
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
    
    @lru_cache(maxsize=1000)
    def _normalize_embedding(self, embedding_tuple: tuple) -> np.ndarray:
        """Cached embedding normalization for performance"""
        embedding = np.array(embedding_tuple, dtype=np.float32)
        return embedding / (np.linalg.norm(embedding) + 1e-8)
    
    def _calculate_similarity_batch(self, query_embedding: np.ndarray, stored_embeddings: List[np.ndarray]) -> np.ndarray:
        """Vectorized batch similarity calculation for optimal performance"""
        try:
            # Normalize query embedding
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            
            # Stack stored embeddings for vectorized computation
            stored_matrix = np.vstack(stored_embeddings)
            stored_norms = stored_matrix / (np.linalg.norm(stored_matrix, axis=1, keepdims=True) + 1e-8)
            
            # Vectorized cosine similarity
            similarities = np.dot(stored_norms, query_norm)
            return similarities
        except Exception as e:
            print(f"Error in batch similarity calculation: {e}")
            return np.zeros(len(stored_embeddings))
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Single similarity calculation with caching"""
        try:
            # Use cached normalization if available
            if self.config['enable_caching']:
                emb1_tuple = tuple(embedding1.astype(np.float32))
                emb2_tuple = tuple(embedding2.astype(np.float32))
                
                embedding1_norm = self._normalize_embedding(emb1_tuple)
                embedding2_norm = self._normalize_embedding(emb2_tuple)
            else:
                embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
                embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
            
            similarity = np.dot(embedding1_norm, embedding2_norm)
            return float(similarity)
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def recognize_face_from_embedding(self, face_embedding: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """Optimized face recognition with batch processing and caching"""
        try:
            if not self.authorized_persons:
                return "Unknown", 0.0, {}
            
            # Extract data for batch processing
            person_ids = list(self.authorized_persons.keys())
            embeddings = [person_data['encoding'] for person_data in self.authorized_persons.values()]
            
            # Batch similarity calculation for optimal performance
            similarities = self._calculate_similarity_batch(face_embedding, embeddings)
            
            # Find best match above threshold
            best_idx = np.argmax(similarities)
            best_confidence = float(similarities[best_idx])
            
            if best_confidence >= self.recognition_threshold:
                best_person_id = person_ids[best_idx]
                person_data = self.authorized_persons[best_person_id]
                
                # Update stats
                self._processing_stats['cache_hits'] += 1
                
                return person_data['name'], best_confidence, {
                    'employee_id': person_data['employee_id'],
                    'department': person_data['department'],
                    'access_level': person_data['access_level'],
                    'status': person_data['status'],
                    'person_id': best_person_id
                }
            else:
                self._processing_stats['cache_misses'] += 1
                return "Unknown", 0.0, {}
        
        except Exception as e:
            print(f"Error in optimized face recognition: {e}")
            return "Error", 0.0, {}
    
    def _upload_image_to_azure(self, image: np.ndarray, filename: str) -> str:
        """Upload unauthorized person image to Azure storage with date-based folder structure"""
        try:
            # Convert image to bytes in memory
            success, buffer = cv2.imencode('.jpg', image)
            if not success:
                raise ValueError("Failed to encode image")
            
            image_bytes = buffer.tobytes()
            
            # Create date-based folder structure
            today = datetime.now().strftime("%Y-%m-%d")
            blob_path = f"{self.images_folder}{today}/{filename}"
            
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_path
            )
            
            blob_client.upload_blob(image_bytes, overwrite=True)
            
            # Return the Azure blob URL
            blob_url = f"https://{blob_client.account_name}.blob.core.windows.net/{self.container_name}/{blob_path}"
            print(f"📤 Uploaded unauthorized image to Azure: {today}/{filename}")
            return blob_url
            
        except Exception as e:
            print(f"❌ Error uploading image to Azure: {e}")
            return ""
    
    def _log_prediction_to_azure(self, log_entry: Dict[str, Any]) -> bool:
        """Log face recognition prediction to Azure in JSON format with daily files"""
        try:
            # Create daily log file name
            today = datetime.now().strftime("%Y-%m-%d")
            log_filename = f"detection_log_{today}.json"
            log_blob_path = f"{self.log_blob}{log_filename}"
            
            print(f"🔍 Logging to Azure: {log_blob_path}")  # Debug output
            
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=log_blob_path
            )
            
            # Download existing log data
            try:
                existing_data = blob_client.download_blob().readall().decode('utf-8')
                log_data = json.loads(existing_data)
                print(f"📝 Found existing log with {len(log_data.get('detections', []))} entries")  # Debug
            except (ResourceNotFoundError, json.JSONDecodeError) as e:
                print(f"📝 Creating new log file: {e}")  # Debug
                # Initialize new log structure
                log_data = {
                    "metadata": {
                        "last_updated": datetime.now().isoformat(),
                        "total_detections": 0,
                        "unauthorized_count": 0,  
                        "authorized_count": 0,
                        "log_file": log_filename,
                        "azure_container": self.container_name
                    },
                    "detections": []
                }
            
            # Process each face detection from log_entry
            faces_to_log = log_entry.get('faces', [])
            print(f"📝 Processing {len(faces_to_log)} faces for logging")  # Debug
            
            for face in faces_to_log:
                detection_id = log_data["metadata"]["total_detections"] + 1
                timestamp = datetime.now()
                
                # Determine status and alert level
                status = "AUTHORIZED" if face['authorized'] else "UNAUTHORIZED"
                alert_level = "LOW" if face['authorized'] else "HIGH"
                
                # Create detection entry
                detection_entry = {
                    "id": detection_id,
                    "timestamp": timestamp.isoformat(),
                    "human_time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": status,
                    "person_name": face['name'],
                    "confidence": f"{int(face['confidence'] * 100)}%",
                    "location": "Camera",
                    "alert_level": alert_level,
                    "additional_info": None
                }
                
                log_data["detections"].append(detection_entry)
                print(f"📝 Added detection: {status} - {face['name']} ({face['confidence']:.2f})")  # Debug
                
                # Update counters
                log_data["metadata"]["total_detections"] += 1
                if face['authorized']:
                    log_data["metadata"]["authorized_count"] += 1
                else:
                    log_data["metadata"]["unauthorized_count"] += 1
            
            # Update metadata
            log_data["metadata"]["last_updated"] = datetime.now().isoformat()
            
            # Upload updated log data
            log_json = json.dumps(log_data, indent=2, ensure_ascii=False)
            blob_client.upload_blob(log_json.encode('utf-8'), overwrite=True)
            
            print(f"📝 Successfully logged {len(log_entry.get('faces', []))} detections to Azure: {log_filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error logging to Azure: {e}")
            import traceback
            traceback.print_exc()  # More detailed error info
            return False
    
    def process_image_optimized(self, image: np.ndarray, source_info: str = "") -> Dict[str, Any]:
        """Hardware-optimized image processing with intelligent resource management"""
        start_time = datetime.now()
        print(f"🔍 Processing image with hardware optimization...")
        
        # Intelligent image preprocessing based on hardware capabilities
        original_shape = image.shape[:2]
        processed_image = self._preprocess_image(image)
        
        if processed_image.shape[:2] != original_shape:
            height, width = processed_image.shape[:2]
            orig_h, orig_w = original_shape
            print(f"📏 Optimized image size: {orig_w}x{orig_h} → {width}x{height}")
        
        # Face detection with performance monitoring
        faces = self.face_app.get(processed_image)
        
        # Intelligent face filtering based on hardware capacity
        if len(faces) > self.config['max_faces']:
            faces = sorted(faces, key=lambda x: x.det_score, reverse=True)[:self.config['max_faces']]
            print(f"👥 Detected {len(faces)} faces (optimized for hardware capacity)")
        else:
            print(f"👥 Detected {len(faces)} faces")
        
        # Memory cleanup
        if processed_image is not image:
            del processed_image
            gc.collect()
        
        # Process faces with parallel recognition if beneficial
        results = {
            'source': source_info,
            'timestamp': start_time.isoformat(),
            'total_faces': len(faces),
            'authorized_count': 0,
            'unauthorized_count': 0,
            'faces': [],
            'processing_time': 0.0,
            'hardware_info': {
                'detection_size': self.detection_size,
                'max_faces': self.config['max_faces'],
                'threads_used': self.config['thread_pool_size']
            }
        }
        
        # Process faces - use parallel processing if multiple faces and sufficient resources
        if len(faces) > 2 and self.config['thread_pool_size'] > 1:
            face_results = self._process_faces_parallel(faces, image, start_time)
        else:
            face_results = self._process_faces_sequential(faces, image, start_time)
        
        # Aggregate results
        for face_result in face_results:
            if face_result['recognition']['authorized']:
                results['authorized_count'] += 1
            else:
                results['unauthorized_count'] += 1
            results['faces'].append(face_result)
        
        # Update performance statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        results['processing_time'] = processing_time
        self._update_processing_stats(processing_time)
        
        # Log to Azure with optimization
        self._log_prediction_to_azure_optimized({
            'source': source_info,
            'total_faces': results['total_faces'],
            'authorized_count': results['authorized_count'],
            'unauthorized_count': results['unauthorized_count'],
            'processing_method': 'Hardware Optimized InsightFace + Azure Cloud',
            'faces': [{
                'face_id': face['face_id'],
                'name': face['recognition']['name'],
                'confidence': face['recognition']['confidence'],
                'authorized': face['recognition']['authorized'],
                'age': face['age'],
                'gender': face['gender']
            } for face in results['faces']]
        })
        
        return results
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Intelligent image preprocessing based on hardware capabilities"""
        height, width = image.shape[:2]
        
        # Calculate optimal size based on memory constraints
        max_pixels = int(system_optimizer.memory_gb * 300000)  # ~300k pixels per GB
        current_pixels = height * width
        
        if current_pixels > max_pixels:
            scale = np.sqrt(max_pixels / current_pixels)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Ensure minimum size for face detection
            new_width = max(320, new_width)
            new_height = max(240, new_height)
            
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return image
    
    def _process_faces_parallel(self, faces: List, image: np.ndarray, start_time: datetime) -> List[Dict]:
        """Parallel face processing for multiple faces"""
        def process_single_face(face_data):
            face, index = face_data
            return self._process_single_face(face, index, image, start_time)
        
        # Submit tasks to thread pool
        face_data = [(face, i) for i, face in enumerate(faces)]
        future_results = list(self.executor.map(process_single_face, face_data))
        
        return future_results
    
    def _process_faces_sequential(self, faces: List, image: np.ndarray, start_time: datetime) -> List[Dict]:
        """Sequential face processing for optimal single-face performance"""
        return [self._process_single_face(face, i, image, start_time) for i, face in enumerate(faces)]
    
    def _process_single_face(self, face, index: int, image: np.ndarray, start_time: datetime) -> Dict:
        """Process a single face with optimization"""
        face_id = f"face_{index+1}"
        
        # Extract face information
        bbox = face.bbox.astype(int)
        x, y, x2, y2 = bbox
        w, h = x2 - x, y2 - y
        
        # Get face embedding
        face_embedding = face.normed_embedding
        
        # Recognize face using optimized method
        name, confidence, metadata = self.recognize_face_from_embedding(face_embedding)
        
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
        
        # Handle unauthorized persons with async upload
        if not face_result['recognition']['authorized']:
            face_region = image[y:y2, x:x2]
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"unauthorized_{timestamp}_{face_id}.jpg"
            
            # Async image upload
            future = self.executor.submit(self._upload_image_to_azure, face_region, filename)
            # Store future for later retrieval if needed
            face_result['upload_future'] = future
        
        return face_result
    
    def _update_processing_stats(self, processing_time: float):
        """Update performance statistics"""
        self._processing_stats['total_processed'] += 1
        self._processing_stats['avg_processing_time'] = (
            (self._processing_stats['avg_processing_time'] * (self._processing_stats['total_processed'] - 1) + processing_time) 
            / self._processing_stats['total_processed']
        )
    
    def _log_prediction_to_azure_optimized(self, log_entry: Dict[str, Any]) -> None:
        """Optimized async logging to Azure"""
        if self.config['enable_caching']:
            # Submit to thread pool for async processing
            self.executor.submit(self._log_prediction_to_azure, log_entry)
        else:
            # Immediate logging for memory-constrained systems
            self._log_prediction_to_azure(log_entry)

    
    def get_authorized_persons_count(self) -> int:
        """Get count of loaded authorized persons"""
        return len(self.authorized_persons)
    
    def get_authorized_persons_list(self) -> List[str]:
        """Get list of authorized person names"""
        return [person['name'] for person in self.authorized_persons.values()]
    
    def reload_embeddings_from_azure(self) -> bool:
        """Reload embeddings from Azure with cache clearing"""
        if self.config['enable_caching']:
            self._normalize_embedding.cache_clear()
        return self._load_embeddings_from_azure()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            **self._processing_stats,
            'system_info': {
                'cpu_cores': system_optimizer.cpu_cores,
                'memory_gb': system_optimizer.memory_gb,
                'architecture': 'ARM64' if system_optimizer.is_arm else 'x86_64',
                'optimal_threads': system_optimizer.optimal_threads
            },
            'configuration': self.config
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        gc.collect()

class OptimizedFaceRecognitionSystem:
    """Hardware-optimized face recognition system with intelligent resource management"""
    
    def __init__(self):
        self.face_processor = OptimizedFaceProcessor()
        print(f"🌐 Optimized Face Recognition System initialized")
        print(f"✅ Loaded {self.face_processor.get_authorized_persons_count()} authorized persons from Azure")
        print(f"🔧 Hardware: {system_optimizer.cpu_cores} cores, {system_optimizer.memory_gb:.1f}GB RAM")
        print(f"🚀 Architecture: {'ARM64' if system_optimizer.is_arm else 'x86_64'}")
        print(f"⚡ Optimizations: Dynamic threading, vectorized computation, intelligent caching")
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image file with hardware optimization"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image with memory optimization
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image: {image_path}")
        
        print(f"📁 Processing image: {image_path}")
        
        # Process using hardware-optimized approach
        results = self.face_processor.process_image_optimized(
            image, 
            f"Image: {os.path.basename(image_path)}"
        )
        
        # Immediate memory cleanup
        del image
        gc.collect()
        
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
        print("🔧 Using InsightFace + Azure Cloud Storage (Jetson CPU optimized)")
        print("Press 'q' to quit")
        
        frame_count = 0
        processing_time_avg = 0.0
        
        # Set camera resolution for Jetson optimization
        if self.cloud_processor.jetson_optimized:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for CPU optimization
            print("🎯 Camera optimized for Jetson: 640x480 @ 15fps")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Dynamic frame skip based on Jetson optimization
            frame_skip = self.cloud_processor.frame_skip_webcam if self.cloud_processor.jetson_optimized else 15
            
            if frame_count % frame_skip == 0:
                # Time the processing for performance monitoring
                import time
                start_time = time.time()
                
                # Process with pure cloud approach
                results = self.cloud_processor.process_image_pure_cloud(
                    frame,
                    f"Webcam - Frame {frame_count}"
                )
                
                # Calculate processing time
                processing_time = time.time() - start_time
                processing_time_avg = (processing_time_avg * 0.8) + (processing_time * 0.2)  # Moving average
                
                # Draw results on frame (no local saving)
                self._draw_results_on_frame(frame, results)
                
                # Add performance info for Jetson monitoring
                if self.cloud_processor.jetson_optimized:
                    perf_text = f"CPU: {processing_time:.2f}s avg:{processing_time_avg:.2f}s"
                    cv2.putText(frame, perf_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
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
        # Initialize optimized system
        system = OptimizedFaceRecognitionSystem()
        
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