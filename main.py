#!/usr/bin/env python3
"""
Face Recognition System
Uses InsightFace for accurate processing but stores everything in Azure cloud
No local storage or dependencies beyond the processing engine
"""

import cv2
import numpy as np
import argparse
import os
import multiprocessing
from datetime import datetime, timedelta
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

class ProductionLogManager:
    """
    Comprehensive production-level logging system for Azure with critical alert support
    Designed for email notifications and detailed audit trails
    """
    
    def __init__(self, blob_service_client, container_name: str):
        self.blob_service_client = blob_service_client
        self.container_name = container_name
        self.log_base_path = "production_logs/"
        
        # Log categories and their Azure paths
        self.log_categories = {
            'security': f"{self.log_base_path}security/",
            'authentication': f"{self.log_base_path}authentication/", 
            'system': f"{self.log_base_path}system/",
            'performance': f"{self.log_base_path}performance/",
            'alerts': f"{self.log_base_path}alerts/",
            'audit': f"{self.log_base_path}audit/"
        }
        
        # Alert levels and their criticality
        self.alert_levels = {
            'INFO': {'priority': 1, 'email_alert': False, 'color': 'GREEN'},
            'WARNING': {'priority': 2, 'email_alert': False, 'color': 'YELLOW'},
            'ERROR': {'priority': 3, 'email_alert': True, 'color': 'ORANGE'},
            'CRITICAL': {'priority': 4, 'email_alert': True, 'color': 'RED'}
        }
        
        # Critical events that require immediate alerts
        self.critical_events = {
            'SECURITY_BREACH': 'Multiple unauthorized access attempts detected',
            'SYSTEM_FAILURE': 'Critical system component failure',
            'AUTH_FAILURE': 'Authentication system malfunction',
            'PERFORMANCE_DEGRADATION': 'System performance below acceptable thresholds',
            'UNAUTHORIZED_ACCESS': 'Unauthorized personnel detected in restricted area',
            'SYSTEM_COMPROMISE': 'Potential system security compromise detected'
        }
        
        print(f"[INIT] Production Log Manager initialized with {len(self.log_categories)} categories")
    
    def log_event(self, category: str, level: str, event_type: str, message: str, 
                  metadata: Dict[str, Any] = None, source: str = "SYSTEM") -> bool:
        """
        Log an event to Azure with comprehensive metadata
        
        Args:
            category: Log category (security, authentication, system, performance, alerts, audit)
            level: Alert level (INFO, WARNING, ERROR, CRITICAL)
            event_type: Type of event for categorization
            message: Human-readable message
            metadata: Additional structured data
            source: Source of the event (SYSTEM, USER, API, etc.)
        """
        try:
            if category not in self.log_categories:
                category = 'system'  # Default fallback
                
            if level not in self.alert_levels:
                level = 'INFO'  # Default fallback
                
            timestamp = datetime.now()
            today = timestamp.strftime("%Y-%m-%d")
            
            # Create log entry
            log_entry = {
                'id': self._generate_log_id(),
                'timestamp': timestamp.isoformat(),
                'human_time': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'category': category,
                'level': level,
                'event_type': event_type,
                'message': message,
                'source': source,
                'alert_info': self.alert_levels[level],
                'metadata': metadata or {},
                'system_info': {
                    'hostname': os.getenv('COMPUTERNAME', 'UNKNOWN'),
                    'process_id': os.getpid(),
                    'thread_id': threading.get_ident()
                }
            }
            
            # Add critical event information if applicable
            if event_type in self.critical_events:
                log_entry['critical_alert'] = {
                    'description': self.critical_events[event_type],
                    'requires_immediate_attention': True,
                    'escalation_required': level in ['ERROR', 'CRITICAL']
                }
            
            # Store in category-specific daily log file
            log_filename = f"{category}_log_{today}.json"
            log_path = f"{self.log_categories[category]}{log_filename}"
            
            # Store the log entry
            self._store_log_entry(log_path, log_entry)
            
            # If it's a critical alert, also store in alerts category
            if self.alert_levels[level]['email_alert']:
                alert_filename = f"critical_alerts_{today}.json"
                alert_path = f"{self.log_categories['alerts']}{alert_filename}"
                self._store_log_entry(alert_path, log_entry)
                
                print(f"[ALERT] {level} alert logged: {event_type} - {message}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to log event: {e}")
            return False
    
    def log_security_event(self, event_type: str, message: str, 
                          person_name: str = None, confidence: float = None, 
                          location: str = "Camera", threat_level: str = "MEDIUM") -> bool:
        """Log security-specific events with enhanced metadata"""
        metadata = {
            'person_name': person_name,
            'confidence': confidence,
            'location': location,
            'threat_level': threat_level,
            'requires_investigation': threat_level in ['HIGH', 'CRITICAL']
        }
        
        # Determine alert level based on threat
        if threat_level == 'CRITICAL':
            level = 'CRITICAL'
        elif threat_level == 'HIGH':
            level = 'ERROR'
        elif threat_level == 'MEDIUM':
            level = 'WARNING'
        else:
            level = 'INFO'
            
        return self.log_event('security', level, event_type, message, metadata, 'SECURITY_SYSTEM')
    
    def log_authentication_event(self, event_type: str, person_name: str, 
                                confidence: float, authorized: bool, location: str = "Camera") -> bool:
        """Log authentication events with detailed metadata"""
        metadata = {
            'person_name': person_name,
            'confidence': confidence,
            'authorized': authorized,
            'location': location,
            'access_granted': authorized,
            'risk_score': 1.0 - confidence if not authorized else 0.0
        }
        
        level = 'INFO' if authorized else 'WARNING'
        message = f"{'Authorized' if authorized else 'Unauthorized'} access attempt by {person_name}"
        
        return self.log_event('authentication', level, event_type, message, metadata, 'AUTH_SYSTEM')
    
    def log_system_event(self, event_type: str, message: str, 
                        performance_data: Dict[str, Any] = None) -> bool:
        """Log system events with performance metrics"""
        metadata = {
            'performance_data': performance_data or {},
            'system_health': self._assess_system_health(performance_data)
        }
        
        # Determine level based on system health
        if metadata['system_health'] == 'CRITICAL':
            level = 'CRITICAL'
        elif metadata['system_health'] == 'DEGRADED':
            level = 'ERROR'
        elif metadata['system_health'] == 'WARNING':
            level = 'WARNING'
        else:
            level = 'INFO'
            
        return self.log_event('system', level, event_type, message, metadata, 'SYSTEM_MONITOR')
    
    def get_critical_alerts(self, days: int = 1) -> List[Dict[str, Any]]:
        """Retrieve critical alerts for email notification system"""
        try:
            alerts = []
            for i in range(days):
                date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                alert_filename = f"critical_alerts_{date}.json"
                alert_path = f"{self.log_categories['alerts']}{alert_filename}"
                
                try:
                    blob_client = self.blob_service_client.get_blob_client(
                        container=self.container_name, blob=alert_path
                    )
                    data = blob_client.download_blob().readall().decode('utf-8')
                    day_alerts = json.loads(data)
                    
                    if isinstance(day_alerts.get('entries'), list):
                        alerts.extend(day_alerts['entries'])
                        
                except Exception:
                    continue  # File might not exist for this day
                    
            return sorted(alerts, key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            print(f"[ERROR] Failed to retrieve critical alerts: {e}")
            return []
    
    def _generate_log_id(self) -> str:
        """Generate unique log entry ID"""
        timestamp = datetime.now()
        return f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{timestamp.microsecond}"
    
    def _assess_system_health(self, performance_data: Dict[str, Any]) -> str:
        """Assess system health based on performance data"""
        if not performance_data:
            return 'UNKNOWN'
            
        processing_time = performance_data.get('processing_time', 0)
        memory_usage = performance_data.get('memory_usage_percent', 0)
        cpu_usage = performance_data.get('cpu_usage_percent', 0)
        
        if processing_time > 15 or memory_usage > 90 or cpu_usage > 95:
            return 'CRITICAL'
        elif processing_time > 10 or memory_usage > 80 or cpu_usage > 85:
            return 'DEGRADED'
        elif processing_time > 8 or memory_usage > 70 or cpu_usage > 75:
            return 'WARNING'
        else:
            return 'HEALTHY'
    
    def _store_log_entry(self, log_path: str, log_entry: Dict[str, Any]) -> bool:
        """Store log entry in Azure blob with proper structure"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=log_path
            )
            
            # Try to download existing log file
            try:
                existing_data = blob_client.download_blob().readall().decode('utf-8')
                log_data = json.loads(existing_data)
            except:
                # Create new log file structure
                log_data = {
                    'metadata': {
                        'created': datetime.now().isoformat(),
                        'last_updated': datetime.now().isoformat(),
                        'total_entries': 0,
                        'log_file': log_path.split('/')[-1],
                        'category': log_entry['category'],
                        'version': '2.0'
                    },
                    'entries': []
                }
            
            # Add new entry
            log_data['entries'].append(log_entry)
            log_data['metadata']['last_updated'] = datetime.now().isoformat()
            log_data['metadata']['total_entries'] = len(log_data['entries'])
            
            # Upload updated log
            log_json = json.dumps(log_data, indent=2, ensure_ascii=False)
            blob_client.upload_blob(log_json.encode('utf-8'), overwrite=True)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to store log entry: {e}")
            return False

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
        """Configure environment for maximum speed processing"""
        # Aggressive thread optimization for speed
        os.environ['OMP_NUM_THREADS'] = str(self.cpu_cores)  # Use all cores
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.cpu_cores)
        os.environ['MKL_NUM_THREADS'] = str(self.cpu_cores)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(self.cpu_cores)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.cpu_cores)
        
        # Speed-optimized settings
        os.environ['OPENCV_VIDEOCAPTURE_CACHE_SIZE'] = '0'  # Disable caching for speed
        os.environ['OPENCV_DNN_OPENCL_ALLOW_ALL_DEVICES'] = '1'  # Enable acceleration
        os.environ['OPENCV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2_USE_NEW_API'] = '1'
        
        # Memory allocation optimizations for speed
        os.environ['MALLOC_ARENA_MAX'] = '4'
        os.environ['MALLOC_MMAP_THRESHOLD_'] = '131072'
        
        # ARM-specific optimizations
        if self.is_arm:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU on ARM
            os.environ['OPENCV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2_USE_NEW_API'] = '1'
    
    def get_optimal_detection_size(self) -> tuple:
        """Get optimal detection size for maximum speed"""
        # Smaller detection size for faster processing
        return (320, 320)  # Optimized for speed over maximum accuracy
    
    def get_processing_config(self) -> dict:
        """Get processing configuration optimized for maximum speed"""
        return {
            'thread_pool_size': self.cpu_cores,  # Use all cores for speed
            'batch_size': max(8, int(self.memory_gb * 2)),  # Larger batches for efficiency
            'enable_caching': True,  # Always enable caching for speed
            'parallel_processing': True,  # Enable parallel face processing
            'optimize_memory': False,  # Prioritize speed over memory
            'fast_mode': True  # Enable speed optimizations
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
        
        # Initialize Production Log Manager
        self.production_logger = ProductionLogManager(self.blob_service_client, self.container_name)
        
        # Initialize InsightFace with hardware-optimized settings
        print(f"[INIT] Initializing InsightFace model (Hardware: {system_optimizer.cpu_cores} cores, {system_optimizer.memory_gb:.1f}GB RAM)...")
        
        # Dynamic provider configuration optimized for speed
        providers = ['CPUExecutionProvider']
        provider_options = [{
            'CPUExecutionProvider': {
                'enable_cpu_mem_arena': True,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cpu_mem_limit': int(system_optimizer.memory_gb * 0.6 * 1024 * 1024 * 1024),  # 60% of RAM for speed
                'enable_mem_pattern': False,  # Disable for speed
                'enable_mem_reuse': True,
                'intra_op_num_threads': system_optimizer.cpu_cores,  # Use all cores
                'inter_op_num_threads': system_optimizer.cpu_cores
            }
        }]
        
        self.face_app = FaceAnalysis(providers=providers, allowed_modules=['detection', 'recognition'])  # Only essential modules
        self.face_app.prepare(ctx_id=-1, det_size=self.detection_size)
        
        print(f"[OK] InsightFace initialized - Detection: {self.detection_size}, Threads: {system_optimizer.optimal_threads}")
        print(f"[CONFIG] Batch size: {self.config['batch_size']}, Parallel processing: {self.config['parallel_processing']}")
        
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
        
        print(f"[OPTIMIZATION] Hardware optimizations enabled for sub-second processing")
        print(f"[CONFIG] Thread pool size: {self.config['thread_pool_size']}")
        print(f"[CONFIG] Batch processing: {self.config['batch_size']} faces per batch")
        print(f"[CONFIG] Memory optimization: {self.config['optimize_memory']}")
        print(f"[CONFIG] Caching enabled: {self.config['enable_caching']}")
        
        # Load embeddings with optimization
        self._load_embeddings_from_azure()
    
    def _load_embeddings_from_azure(self) -> bool:
        """Load face embeddings from Azure JSON file - no local caching"""
        try:
            print(f"[LOAD] Loading embeddings from Azure blob: {self.embeddings_blob}")
            
            # Use BlobClient for direct access
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
            
            print(f"[SUCCESS] Successfully loaded {len(self.authorized_persons)} authorized persons from Azure")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error loading embeddings from Azure: {str(e)}")
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
            print(f"[UPLOAD] Uploaded unauthorized image to Azure: {today}/{filename}")
            return blob_url
            
        except Exception as e:
            print(f"[ERROR] Error uploading image to Azure: {e}")
            return ""
    
    def _log_prediction_to_azure(self, log_entry: Dict[str, Any]) -> bool:
        """Log face recognition prediction to Azure in JSON format with daily files"""
        try:
            # Create daily log file name
            today = datetime.now().strftime("%Y-%m-%d")
            log_filename = f"detection_log_{today}.json"
            log_blob_path = f"{self.log_blob}{log_filename}"
            
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=log_blob_path
            )
            
            # Download existing log data
            try:
                existing_data = blob_client.download_blob().readall().decode('utf-8')
                log_data = json.loads(existing_data)
            except (ResourceNotFoundError, json.JSONDecodeError):
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
            for face in log_entry.get('faces', []):
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
            
            print(f"[LOG] Logged {len(log_entry.get('faces', []))} detections to Azure: {log_filename}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error logging to Azure: {e}")
            return False
    
    def process_image_optimized(self, image: np.ndarray, source_info: str = "") -> Dict[str, Any]:
        """Hardware-optimized image processing with intelligent resource management"""
        start_time = datetime.now()
        timing_breakdown = {}
        print(f"[PROCESSING] Processing image with hardware optimization...")
        
        # Phase 1: Image preprocessing
        preprocessing_start = datetime.now()
        original_shape = image.shape[:2]
        processed_image = self._preprocess_image(image)
        preprocessing_time = (datetime.now() - preprocessing_start).total_seconds()
        timing_breakdown['preprocessing'] = preprocessing_time
        
        if processed_image.shape[:2] != original_shape:
            height, width = processed_image.shape[:2]
            orig_h, orig_w = original_shape
            print(f"[OPTIMIZATION] Optimized image size: {orig_w}x{orig_h} → {width}x{height}")
        
        # Phase 2: Face detection
        detection_start = datetime.now()
        faces = self.face_app.get(processed_image)
        detection_time = (datetime.now() - detection_start).total_seconds()
        timing_breakdown['face_detection'] = detection_time
        face_count = len(faces)
        
        print(f"[DETECTION] Processing {face_count} detected faces")
        print(f"[TIMING] Face detection took: {detection_time:.3f}s")
        
        # Memory cleanup
        if processed_image is not image:
            del processed_image
            gc.collect()
        
        # Phase 3: Face recognition setup
        recognition_setup_start = datetime.now()
        results = {
            'source': source_info,
            'timestamp': start_time.isoformat(),
            'total_faces': len(faces),
            'authorized_count': 0,
            'unauthorized_count': 0,
            'faces': [],
            'processing_time': 0.0,
            'timing_breakdown': timing_breakdown,
            'hardware_info': {
                'detection_size': self.detection_size,
                'batch_size': self.config['batch_size'],
                'threads_used': self.config['thread_pool_size']
            }
        }
        
        # Phase 4: Face recognition processing
        face_recognition_start = datetime.now()
        if len(faces) > 1 and self.config['parallel_processing']:
            face_results = self._process_faces_parallel(faces, image, start_time)
        else:
            face_results = self._process_faces_sequential(faces, image, start_time)
        face_recognition_time = (datetime.now() - face_recognition_start).total_seconds()
        timing_breakdown['face_recognition'] = face_recognition_time
        
        print(f"[TIMING] Face recognition took: {face_recognition_time:.3f}s")
        
        # Phase 5: Results aggregation
        aggregation_start = datetime.now()
        for face_result in face_results:
            if face_result['recognition']['authorized']:
                results['authorized_count'] += 1
            else:
                results['unauthorized_count'] += 1
            results['faces'].append(face_result)
        
        aggregation_time = (datetime.now() - aggregation_start).total_seconds()
        timing_breakdown['aggregation'] = aggregation_time
        
        # Phase 6: Logging operations
        logging_start = datetime.now()
        
        # Update performance statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        results['processing_time'] = processing_time
        results['timing_breakdown'] = timing_breakdown
        self._update_processing_stats(processing_time)
        
        # Log to Azure with optimization (original logging)
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
        
        # Production logging system (new comprehensive logging)
        self._log_to_production_system(results, source_info, processing_time, start_time)
        
        logging_time = (datetime.now() - logging_start).total_seconds()
        timing_breakdown['logging'] = logging_time
        
        # Final timing summary
        print(f"[TIMING] === Processing Time Breakdown ===")
        print(f"[TIMING] Preprocessing: {timing_breakdown['preprocessing']:.3f}s")
        print(f"[TIMING] Face Detection: {timing_breakdown['face_detection']:.3f}s")
        print(f"[TIMING] Face Recognition: {timing_breakdown['face_recognition']:.3f}s")
        print(f"[TIMING] Result Aggregation: {timing_breakdown['aggregation']:.3f}s")
        print(f"[TIMING] Logging Operations: {timing_breakdown['logging']:.3f}s")
        print(f"[TIMING] === Total Time: {processing_time:.3f}s ===")
        
        # Calculate face prediction time (detection + recognition)
        face_prediction_time = timing_breakdown['face_detection'] + timing_breakdown['face_recognition']
        print(f"[PREDICTION] Face prediction time: {face_prediction_time:.3f}s ({face_count} faces)")
        if face_count > 0:
            print(f"[PREDICTION] Average per face: {face_prediction_time/face_count:.3f}s")
        
        return results
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Minimal preprocessing for maximum speed"""
        if self.config['fast_mode']:
            # Skip resizing for speed unless absolutely necessary
            height, width = image.shape[:2]
            if width > 2048 or height > 1536:  # Only resize very large images
                scale = min(2048/width, 1536/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)  # Faster interpolation
            return image
        
        # Original preprocessing logic for compatibility
        height, width = image.shape[:2]
        if self.config['optimize_memory'] and (width > 1920 or height > 1080):
            max_dimension = 1920 if system_optimizer.memory_gb >= 4 else 1280
            if max(width, height) > max_dimension:
                scale = max_dimension / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return image
    
    def _process_faces_parallel(self, faces: List, image: np.ndarray, start_time: datetime) -> List[Dict]:
        """Optimized parallel face processing with intelligent batching"""
        batch_size = self.config['batch_size']
        
        # For small face counts, use sequential processing to avoid thread overhead
        if len(faces) <= 4:
            return [self._process_single_face(face, i, image, start_time) for i, face in enumerate(faces)]
        
        # For moderate face counts, use direct parallel processing
        if len(faces) <= batch_size:
            def process_single_face_wrapper(face_data):
                face, index = face_data
                return self._process_single_face(face, index, image, start_time)
            
            face_data = [(face, i) for i, face in enumerate(faces)]
            return list(self.executor.map(process_single_face_wrapper, face_data))
        
        # For large face counts, use batch processing
        def process_face_batch(batch_data):
            batch_faces, start_index = batch_data
            return [self._process_single_face(face, start_index + i, image, start_time) 
                   for i, face in enumerate(batch_faces)]
        
        # Create smaller, more efficient batches
        optimal_batch_size = max(2, len(faces) // self.config['thread_pool_size'])
        batches = []
        for i in range(0, len(faces), optimal_batch_size):
            batch = faces[i:i + optimal_batch_size]
            batches.append((batch, i))
        
        # Process batches in parallel
        batch_results = list(self.executor.map(process_face_batch, batches))
        
        # Flatten results
        return [face_result for batch_result in batch_results for face_result in batch_result]
    
    def _process_faces_sequential(self, faces: List, image: np.ndarray, start_time: datetime) -> List[Dict]:
        """Sequential face processing for optimal single-face performance"""
        return [self._process_single_face(face, i, image, start_time) for i, face in enumerate(faces)]
    
    def _process_single_face(self, face, index: int, image: np.ndarray, start_time: datetime) -> Dict:
        """Process a single face with optimization and timing"""
        face_processing_start = datetime.now()
        face_id = f"face_{index+1}"
        
        # Phase 1: Extract face information
        bbox_start = datetime.now()
        bbox = face.bbox.astype(int)
        x, y, x2, y2 = bbox
        w, h = x2 - x, y2 - y
        bbox_time = (datetime.now() - bbox_start).total_seconds()
        
        # Phase 2: Get face embedding
        embedding_start = datetime.now()
        face_embedding = face.normed_embedding
        embedding_time = (datetime.now() - embedding_start).total_seconds()
        
        # Phase 3: Recognize face using optimized method
        recognition_start = datetime.now()
        name, confidence, metadata = self.recognize_face_from_embedding(face_embedding)
        recognition_time = (datetime.now() - recognition_start).total_seconds()
        
        # Calculate total processing time for this face
        face_total_time = (datetime.now() - face_processing_start).total_seconds()
        
        # Prepare face result
        face_result = {
            'face_id': face_id,
            'recognition': {
                'name': name,
                'confidence': confidence,
                'authorized': name != "Unknown" and confidence >= self.recognition_threshold
            },
            'metadata': metadata,
            'timing': {
                'bbox_extraction': bbox_time,
                'embedding_generation': embedding_time,
                'recognition_matching': recognition_time,
                'total_face_time': face_total_time
            },
            'bbox': [x, y, w, h],
            'age': int(face.age) if hasattr(face, 'age') and face.age is not None else 0,
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
    
    def _log_to_production_system(self, results: Dict[str, Any], source_info: str, 
                                 processing_time: float, start_time: datetime) -> None:
        """Comprehensive production logging with critical alert support"""
        try:
            # Log system performance event
            performance_data = {
                'processing_time': processing_time,
                'faces_detected': results['total_faces'],
                'authorized_count': results['authorized_count'], 
                'unauthorized_count': results['unauthorized_count'],
                'memory_usage_percent': self._get_memory_usage(),
                'threads_used': self.config['thread_pool_size'],
                'detection_size': self.detection_size
            }
            
            self.production_logger.log_system_event(
                'FACE_PROCESSING_COMPLETED',
                f"Processed {results['total_faces']} faces in {processing_time:.2f}s from {source_info}",
                performance_data
            )
            
            # Log authentication events for each face
            for face in results['faces']:
                self.production_logger.log_authentication_event(
                    'FACE_RECOGNITION_ATTEMPT',
                    face['recognition']['name'],
                    face['recognition']['confidence'],
                    face['recognition']['authorized']
                )
                
                # Log security events for unauthorized access
                if not face['recognition']['authorized']:
                    threat_level = self._assess_threat_level(face)
                    self.production_logger.log_security_event(
                        'UNAUTHORIZED_ACCESS_DETECTED',
                        f"Unauthorized person detected: {face['recognition']['name']} (confidence: {face['recognition']['confidence']:.2f})",
                        face['recognition']['name'],
                        face['recognition']['confidence'],
                        threat_level=threat_level
                    )
            
            # Check for critical security scenarios
            self._check_critical_security_scenarios(results, source_info)
            
        except Exception as e:
            print(f"[ERROR] Production logging failed: {e}")
            # Log the logging failure itself
            try:
                self.production_logger.log_system_event(
                    'LOGGING_SYSTEM_ERROR',
                    f"Production logging system failure: {e}",
                    {'error_type': type(e).__name__, 'error_message': str(e)}
                )
            except:
                pass  # Prevent recursive errors
    
    def _assess_threat_level(self, face: Dict[str, Any]) -> str:
        """Assess threat level based on face detection confidence and other factors"""
        confidence = face['recognition']['confidence']
        detection_confidence = face.get('detection_confidence', 0.0)
        
        # Unknown person with high detection confidence = higher threat
        if confidence == 0.0 and detection_confidence > 0.8:
            return 'HIGH'
        elif confidence == 0.0 and detection_confidence > 0.6:
            return 'MEDIUM'
        elif confidence < 0.3:  # Low confidence on known person
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _check_critical_security_scenarios(self, results: Dict[str, Any], source_info: str) -> None:
        """Check for critical security scenarios that require immediate alerts"""
        unauthorized_count = results['unauthorized_count']
        total_faces = results['total_faces']
        
        # Multiple unauthorized persons detected
        if unauthorized_count >= 3:
            self.production_logger.log_event(
                'security', 'CRITICAL', 'SECURITY_BREACH',
                f"Multiple unauthorized persons detected: {unauthorized_count} out of {total_faces} faces",
                {
                    'unauthorized_count': unauthorized_count,
                    'total_faces': total_faces,
                    'source': source_info,
                    'breach_ratio': unauthorized_count / total_faces if total_faces > 0 else 0,
                    'immediate_action_required': True
                },
                'SECURITY_SYSTEM'
            )
        
        # High ratio of unauthorized to total faces
        elif total_faces > 0 and (unauthorized_count / total_faces) > 0.7:
            self.production_logger.log_event(
                'security', 'ERROR', 'HIGH_UNAUTHORIZED_RATIO',
                f"High unauthorized access ratio: {unauthorized_count}/{total_faces} ({(unauthorized_count/total_faces)*100:.1f}%)",
                {
                    'unauthorized_ratio': unauthorized_count / total_faces,
                    'source': source_info,
                    'requires_investigation': True
                },
                'SECURITY_SYSTEM'
            )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage (simplified for cross-platform)"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            # Fallback estimation based on processing stats
            return min(80.0, 50.0 + (self._processing_stats['avg_processing_time'] * 5))
    
    def get_critical_alerts_for_email(self, days: int = 1) -> List[Dict[str, Any]]:
        """Get critical alerts formatted for email notifications"""
        return self.production_logger.get_critical_alerts(days)

    
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
        print(f"[SYSTEM] Optimized Face Recognition System initialized")
        print(f"[STATUS] Loaded {self.face_processor.get_authorized_persons_count()} authorized persons from Azure")
        print(f"[HARDWARE] Hardware: {system_optimizer.cpu_cores} cores, {system_optimizer.memory_gb:.1f}GB RAM")
        print(f"[HARDWARE] Architecture: {'ARM64' if system_optimizer.is_arm else 'x86_64'}")
        print(f"[OPTIMIZATION] Optimizations: Dynamic threading, vectorized computation, intelligent caching")
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image file with hardware optimization"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image with memory optimization
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image: {image_path}")
        
        print(f"[PROCESSING] Processing image: {image_path}")
        
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
        
        print(f"[PROCESSING] Processing video: {video_path}")
        frame_count = 0
        total_results = {
            'source': f"Video: {os.path.basename(video_path)}",
            'total_frames_processed': 0,
            'total_faces_detected': 0,
            'authorized_total': 0,
            'unauthorized_total': 0,
            'frame_results': []
        }
        
        # Process frames efficiently for dedicated processing system
        frame_skip = max(1, int(30 / system_optimizer.memory_gb))  # Adaptive frame processing based on resources
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frames based on available resources
            if frame_count % frame_skip != 0:
                continue
            
            print(f"[PROCESSING] Processing frame {frame_count}")
            
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
        
        print(" Starting pure cloud webcam processing...")
        print(" Using InsightFace + Azure Cloud Storage (Jetson CPU optimized)")
        print("Press 'q' to quit")
        
        frame_count = 0
        processing_time_avg = 0.0
        
        # Set camera resolution for Jetson optimization
        if self.cloud_processor.jetson_optimized:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for CPU optimization
            print("[OPTIMIZATION] Camera optimized for Jetson: 640x480 @ 15fps")
        
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
                label = f"[OK] {name} ({confidence:.2f})"
            else:
                label = f"[UNAUTH] UNAUTHORIZED ({confidence:.2f})"
            
            # Add cloud indicator
            cv2.putText(frame, "[CLOUD]", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    parser = argparse.ArgumentParser(description='Pure Cloud Face Recognition System')
    parser.add_argument('--mode', choices=['image', 'video', 'webcam'], required=True,
                       help='Processing mode')
    parser.add_argument('--input', type=str, help='Input file path (for image/video modes)')
    parser.add_argument('--output', type=str, help='Output file path for results (optional)')
    parser.add_argument('--timing', action='store_true', help='Show detailed timing breakdown')
    
    args = parser.parse_args()
    
    try:
        # Initialize optimized system
        system = OptimizedFaceRecognitionSystem()
        
        # Process based on mode
        if args.mode == 'image':
            if not args.input:
                print("[ERROR] --input required for image mode")
                return
            
            results = system.process_image(args.input)
            
            # Print comprehensive results
            print(f"\n[RESULTS] === Pure Cloud Processing Results ===")
            print(f"[INFO] Source: {results['source']}")
            print(f"[INFO] Total faces: {results['total_faces']}")
            print(f"[INFO] Authorized: {results['authorized_count']}")
            print(f"[INFO] Unauthorized: {results['unauthorized_count']}")
            print(f"[INFO] All data stored in Azure Cloud")
            
            # Show detailed results
            print(f"\n[DETAILS] Detailed Recognition Results:")
            authorized_names = []
            for i, face in enumerate(results['faces']):
                status_icon = "[OK]" if face['recognition']['authorized'] else "[UNAUTH]"
                auth_status = "AUTHORIZED" if face['recognition']['authorized'] else "UNAUTHORIZED"
                
                print(f"\n{status_icon} Face {i+1}: {auth_status}")
                print(f"   Name: {face['recognition']['name']}")
                print(f"   Confidence: {face['recognition']['confidence']:.3f}")
                print(f"   Detection: {face['detection_confidence']:.3f}")
                print(f"   Age: {face['age']}, Gender: {face['gender']}")
                
                if face['recognition']['authorized']:
                    authorized_names.append(face['recognition']['name'])
                    if 'metadata' in face and face['metadata']:
                        metadata = face['metadata']
                        print(f"   Employee ID: {metadata.get('employee_id', 'N/A')}")
                        print(f"   Department: {metadata.get('department', 'N/A')}")
                        print(f"   Access Level: {metadata.get('access_level', 'N/A')}")
                
                if not face['recognition']['authorized'] and 'unauthorized_image_url' in face:
                    print(f"   Image uploaded to: {face['unauthorized_image_url']}")
            
            if authorized_names:
                print(f"\n[SUCCESS] Authorized personnel detected: {', '.join(set(authorized_names))}")
            
            # Show detailed timing if requested
            if args.timing and 'timing_breakdown' in results:
                timing = results['timing_breakdown']
                print(f"\n[TIMING] === Detailed Performance Analysis ===")
                print(f"[TIMING] Total Processing Time: {results['processing_time']:.3f}s")
                print(f"[TIMING] Image Preprocessing: {timing.get('preprocessing', 0):.3f}s")
                print(f"[TIMING] Face Detection: {timing.get('face_detection', 0):.3f}s")
                print(f"[TIMING] Face Recognition: {timing.get('face_recognition', 0):.3f}s")
                print(f"[TIMING] Result Aggregation: {timing.get('aggregation', 0):.3f}s")
                print(f"[TIMING] Logging Operations: {timing.get('logging', 0):.3f}s")
                
                # Face prediction time analysis
                face_prediction_time = timing.get('face_detection', 0) + timing.get('face_recognition', 0)
                print(f"\n[PREDICTION] === Face Prediction Analysis ===")
                print(f"[PREDICTION] Total Face Prediction Time: {face_prediction_time:.3f}s")
                print(f"[PREDICTION] Faces Processed: {results['total_faces']}")
                if results['total_faces'] > 0:
                    avg_per_face = face_prediction_time / results['total_faces']
                    print(f"[PREDICTION] Average per Face: {avg_per_face:.3f}s")
                
                # Show individual face timings if available
                if results['faces'] and 'timing' in results['faces'][0]:
                    print(f"\n[FACE_TIMING] === Individual Face Processing Times ===")
                    total_bbox_time = 0
                    total_embedding_time = 0
                    total_recognition_time = 0
                    
                    for i, face in enumerate(results['faces'][:5], 1):  # Show first 5 faces
                        if 'timing' in face:
                            timing_data = face['timing']
                            print(f"[FACE_TIMING] Face {i}: Total {timing_data.get('total_face_time', 0):.3f}s")
                            print(f"   - Bbox Extraction: {timing_data.get('bbox_extraction', 0):.3f}s")
                            print(f"   - Embedding Generation: {timing_data.get('embedding_generation', 0):.3f}s")
                            print(f"   - Recognition Matching: {timing_data.get('recognition_matching', 0):.3f}s")
                            
                            total_bbox_time += timing_data.get('bbox_extraction', 0)
                            total_embedding_time += timing_data.get('embedding_generation', 0)
                            total_recognition_time += timing_data.get('recognition_matching', 0)
                    
                    if results['total_faces'] > 5:
                        print(f"[FACE_TIMING] ... (showing first 5 of {results['total_faces']} faces)")
                    
                    # Show averages
                    face_count = min(5, results['total_faces'])
                    if face_count > 0:
                        print(f"\n[FACE_TIMING] === Average Times (first {face_count} faces) ===")
                        print(f"[FACE_TIMING] Avg Bbox Extraction: {total_bbox_time/face_count:.3f}s")
                        print(f"[FACE_TIMING] Avg Embedding Generation: {total_embedding_time/face_count:.3f}s")
                        print(f"[FACE_TIMING] Avg Recognition Matching: {total_recognition_time/face_count:.3f}s")
                
                # Performance insights
                overhead_time = results['processing_time'] - face_prediction_time
                print(f"\n[ANALYSIS] === Performance Insights ===")
                print(f"[ANALYSIS] Pure Face Prediction: {face_prediction_time:.3f}s ({face_prediction_time/results['processing_time']*100:.1f}%)")
                print(f"[ANALYSIS] System Overhead: {overhead_time:.3f}s ({overhead_time/results['processing_time']*100:.1f}%)")
                
                if face_prediction_time > 5:
                    print(f"[RECOMMENDATION] Face prediction time is high. Consider:")
                    print(f"   - Reducing detection size (currently {results['hardware_info']['detection_size']})")
                    print(f"   - Using faster recognition models")
                    print(f"   - Optimizing parallel processing")
            
            # Save results if output specified (only metadata, no local images)
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\n[SAVE] Results metadata saved to: {args.output}")
                print(f"[INFO] All images and logs stored in Azure Cloud")
        
        elif args.mode == 'video':
            if not args.input:
                print("[ERROR] --input required for video mode")
                return
            
            results = system.process_video(args.input)
            
            print(f"\n[RESULTS] === Pure Cloud Video Processing Results ===")
            print(f"[INFO] Source: {results['source']}")
            print(f"[INFO] Frames processed: {results['total_frames_processed']}")
            print(f"[INFO] Total faces detected: {results['total_faces_detected']}")
            print(f"[INFO] Total authorized: {results['authorized_total']}")
            print(f"[INFO] Total unauthorized: {results['unauthorized_total']}")
            print(f"[INFO] All data stored in Azure Cloud")
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\n[SAVE] Results metadata saved to: {args.output}")
        
        elif args.mode == 'webcam':
            system.process_webcam()
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

def format_critical_alerts_for_email(processor: OptimizedFaceProcessor, days: int = 1) -> Dict[str, Any]:
    """
    Format critical alerts for email notification system
    Returns structured data ready for email templates
    """
    try:
        alerts = processor.get_critical_alerts_for_email(days)
        
        if not alerts:
            return {
                'has_alerts': False,
                'summary': 'No critical alerts in the specified period.',
                'alert_count': 0
            }
        
        # Categorize alerts by severity and type
        critical_alerts = [a for a in alerts if a['level'] == 'CRITICAL']
        error_alerts = [a for a in alerts if a['level'] == 'ERROR']
        
        # Group by event type for summary
        event_summary = {}
        for alert in alerts:
            event_type = alert['event_type']
            if event_type not in event_summary:
                event_summary[event_type] = {
                    'count': 0,
                    'latest': alert['human_time'],
                    'severity': alert['level']
                }
            event_summary[event_type]['count'] += 1
        
        # Create email-ready format
        return {
            'has_alerts': True,
            'alert_count': len(alerts),
            'critical_count': len(critical_alerts),
            'error_count': len(error_alerts),
            'time_period': f"Last {days} day(s)",
            'summary': f"Found {len(alerts)} critical alerts requiring attention",
            'top_alerts': alerts[:10],  # Most recent 10 alerts
            'event_summary': event_summary,
            'recommendations': _generate_alert_recommendations(alerts),
            'formatted_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'system_status': _assess_overall_system_status(alerts)
        }
        
    except Exception as e:
        return {
            'has_alerts': True,
            'error': True,
            'message': f"Failed to retrieve alerts: {e}",
            'alert_count': 0
        }

def _generate_alert_recommendations(alerts: List[Dict[str, Any]]) -> List[str]:
    """Generate actionable recommendations based on alert patterns"""
    recommendations = []
    
    # Count different types of alerts
    security_alerts = len([a for a in alerts if a['category'] == 'security'])
    auth_alerts = len([a for a in alerts if a['category'] == 'authentication'])
    system_alerts = len([a for a in alerts if a['category'] == 'system'])
    
    if security_alerts > 5:
        recommendations.append("🚨 High number of security alerts detected. Consider reviewing access control policies.")
    
    if auth_alerts > 10:
        recommendations.append("👤 Frequent authentication failures. Review authorized personnel database.")
    
    if system_alerts > 3:
        recommendations.append("⚙️ System performance issues detected. Consider hardware upgrade or optimization.")
    
    # Check for critical events
    critical_events = [a for a in alerts if a.get('critical_alert', {}).get('requires_immediate_attention')]
    if critical_events:
        recommendations.append("🔴 Immediate attention required for critical security events.")
    
    if not recommendations:
        recommendations.append("✅ System operating within normal parameters.")
    
    return recommendations

def _assess_overall_system_status(alerts: List[Dict[str, Any]]) -> str:
    """Assess overall system health based on recent alerts"""
    if not alerts:
        return "HEALTHY"
    
    critical_count = len([a for a in alerts if a['level'] == 'CRITICAL'])
    error_count = len([a for a in alerts if a['level'] == 'ERROR'])
    
    if critical_count > 0:
        return "CRITICAL"
    elif error_count > 5:
        return "DEGRADED"
    elif error_count > 0:
        return "WARNING"
    else:
        return "HEALTHY"

if __name__ == "__main__":
    main()