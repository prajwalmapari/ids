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

def get_geolocation():
    """Get current geolocation with dynamic detection - no hardcoded coordinates"""
    
    # Method 1: Check for precise coordinates in environment (user-provided only)
    precise_coords = os.getenv('PRECISE_COORDINATES')
    if precise_coords:
        try:
            lat_str, lon_str = precise_coords.split(',')
            lat, lon = float(lat_str.strip()), float(lon_str.strip())
            return f"{lat:.6f}, {lon:.6f}"
        except Exception:
            pass
    
    # Method 2: Try system GPS (real GPS hardware - most accurate)
    try:
        import subprocess
        # Try gpsd first (GPS daemon)
        result = subprocess.run(['gpspipe', '-w', '-n', '5'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            import json
            lines = result.stdout.strip().split('\n')
            for line in lines:
                try:
                    gps_data = json.loads(line)
                    if gps_data.get('class') == 'TPV' and 'lat' in gps_data and 'lon' in gps_data:
                        lat, lon = gps_data['lat'], gps_data['lon']
                        # Validate GPS coordinates are reasonable
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            print(f"🛰️ GPS Hardware: {lat:.6f}, {lon:.6f}")
                            return f"{lat:.6f}, {lon:.6f}"
                except Exception:
                    continue
    except Exception:
        pass
    
    # Method 3: Try Android location services (if available)
    try:
        import subprocess
        # Try termux-location for Android environments
        result = subprocess.run(['termux-location'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            import json
            location_data = json.loads(result.stdout)
            lat = location_data.get('latitude')
            lon = location_data.get('longitude')
            if lat and lon and -90 <= lat <= 90 and -180 <= lon <= 180:
                print(f"📱 Android GPS: {lat:.6f}, {lon:.6f}")
                return f"{lat:.6f}, {lon:.6f}"
    except Exception:
        pass
    
    # Method 4: Try geocoder library with GPS preference
    try:
        import geocoder
        
        # Try GPS-based methods first (requires GPS hardware)
        for method in ['osm', 'google']:
            try:
                g = geocoder.get('me', method=method)
                if g.ok and g.latlng:
                    lat, lon = g.latlng
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        print(f"🌐 Geocoder-{method}: {lat:.6f}, {lon:.6f}")
                        return f"{lat:.6f}, {lon:.6f}"
            except Exception:
                continue
        
        # Fallback to IP-based (less accurate but available)
        g = geocoder.ip('me')
        if g.ok and g.latlng:
            lat, lon = g.latlng
            print(f"🌍 IP-based location: {lat:.6f}, {lon:.6f}")
            return f"{lat:.6f}, {lon:.6f}"
                
    except ImportError:
        print("⚠️ Geocoder library not available")
        pass
    except Exception:
        pass
    
    # Method 5: Web-based geolocation services (IP-based - dynamic detection)
    services = [
        ('ip-api.com', 'http://ip-api.com/json/'),
        ('ipinfo.io', 'https://ipinfo.io/json'),
        ('ipapi.co', 'http://ipapi.co/json/')
    ]
    
    import requests
    for service_name, service_url in services:
        try:
            response = requests.get(service_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                # Handle different response formats
                lat = data.get('latitude') or data.get('lat')
                lon = data.get('longitude') or data.get('lon')
                
                # Handle ipinfo.io format
                if not lat and 'loc' in data:
                    coords = data['loc'].split(',')
                    if len(coords) == 2:
                        lat, lon = float(coords[0]), float(coords[1])
                
                if lat and lon:
                    print(f"🌐 {service_name}: {lat:.6f}, {lon:.6f}")
                    return f"{lat:.6f}, {lon:.6f}"
        except Exception:
            continue
    
    # Method 6: Check for user-defined static coordinates (not hardcoded)
    static_coords = os.getenv('STATIC_COORDINATES')
    if static_coords:
        try:
            lat_str, lon_str = static_coords.split(',')
            lat, lon = float(lat_str.strip()), float(lon_str.strip())
            print(f"📍 User-defined static: {lat:.6f}, {lon:.6f}")
            return f"{lat:.6f}, {lon:.6f}"
        except Exception:
            pass
    
    # No hardcoded fallback - return error state
    print("❌ No location detection methods succeeded")
    print("💡 Suggestions:")
    print("  - Check internet connection for IP-based location")
    print("  - Install GPS hardware for precise location")
    print("  - Set PRECISE_COORDINATES environment variable")
    return "0.000000, 0.000000"

def add_timestamp_overlay(image, timestamp_str=None, confidence=None, person_id=None, location=None):
    """Add timestamp and geolocation overlay to image before Azure upload"""
    try:
        # Create a copy of the image to avoid modifying the original
        img_with_overlay = image.copy()
        
        # Get image dimensions
        height, width = img_with_overlay.shape[:2]
        
        # Generate timestamp if not provided
        if timestamp_str is None:
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get geolocation if not provided
        if location is None:
            location = get_geolocation()
        
        # Configure smaller text properties based on image size
        font_scale = max(0.2, min(width/400, height/400))  # Smaller scale factor
        thickness = max(1, int(font_scale * 1.5))
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Colors (BGR format)
        bg_color = (0, 0, 0)       # Black background
        text_color = (0, 255, 255)  # Yellow text
        border_color = (0, 0, 255)  # Red border
        
        # Prepare text lines
        timestamp_text = timestamp_str
        location_text = location if location else "0.0000, 0.0000"
        
        # Calculate line height
        line_height = int(15 * font_scale)
        
        # Position for timestamp (top-left)
        timestamp_y = line_height + 5
        
        # Get timestamp text size
        (ts_width, ts_height), ts_baseline = cv2.getTextSize(timestamp_text, font, font_scale, thickness)
        
        # Draw timestamp background and border
        cv2.rectangle(img_with_overlay, 
                     (2, timestamp_y - ts_height - 2), 
                     (ts_width + 6, timestamp_y + ts_baseline + 2), 
                     bg_color, -1)
        cv2.rectangle(img_with_overlay, 
                     (2, timestamp_y - ts_height - 2), 
                     (ts_width + 6, timestamp_y + ts_baseline + 2), 
                     border_color, 1)
        
        # Draw timestamp text
        cv2.putText(img_with_overlay, timestamp_text, (4, timestamp_y), font, font_scale, text_color, thickness)
        
        # Position for location (bottom-left)
        location_y = height - 8  # 8 pixels from bottom
        
        # Get location text size
        (loc_width, loc_height), loc_baseline = cv2.getTextSize(location_text, font, font_scale, thickness)
        
        # Draw location background and border
        cv2.rectangle(img_with_overlay, 
                     (2, location_y - loc_height - 2), 
                     (loc_width + 6, location_y + loc_baseline + 2), 
                     bg_color, -1)
        cv2.rectangle(img_with_overlay, 
                     (2, location_y - loc_height - 2), 
                     (loc_width + 6, location_y + loc_baseline + 2), 
                     border_color, 1)
        
        # Draw location text
        cv2.putText(img_with_overlay, location_text, (4, location_y), font, font_scale, text_color, thickness)
        
        return img_with_overlay
        
    except Exception as e:
        print(f"⚠️ Error adding timestamp overlay: {e}")
        return image  # Return original image if overlay fails

class SystemOptimizer:
    """Root-level system optimization based on hardware detection with 6-core optimization"""
    
    def __init__(self):
        self.cpu_cores = multiprocessing.cpu_count()
        self.memory_gb = self._get_available_memory()
        self.is_arm = self._detect_arm_architecture()
        self.gpu_info = self._detect_gpu()
        self.optimal_threads = self._calculate_optimal_threads()
        self.setup_environment()
    
    def _get_available_memory(self) -> float:
        """Get available system memory in GB with cross-platform support"""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            try:
                # Linux fallback
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
            import platform
            arch = platform.machine().lower()
            if 'arm' in arch or 'aarch64' in arch:
                return True
            # Fallback for Linux systems
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read().lower()
                return any(arch in content for arch in ['arm', 'aarch64', 'cortex'])
        except:
            return False
    
    def _detect_gpu(self) -> dict:
        """Detect available GPU and configure for 4GB limit"""
        gpu_info = {
            'available': False,
            'memory_gb': 0,
            'provider': None,
            'device_id': 0
        }
        
        print("🔍 Detecting GPU...")
        
        try:
            # First, check if ONNX Runtime supports CUDA
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            print(f"🔧 Available ONNX providers: {available_providers}")
            
            if 'CUDAExecutionProvider' not in available_providers:
                print("❌ CUDAExecutionProvider not available in ONNX Runtime")
                print("💡 You may need to install onnxruntime-gpu: pip install onnxruntime-gpu")
                return gpu_info
            
            # If CUDA provider is available, we can use GPU
            print("✅ CUDAExecutionProvider detected - GPU support available")
            
            # Try to get more detailed GPU info
            device_name = "Unknown NVIDIA GPU"
            total_memory = 8.0  # Default assumption
            
            # Try NVIDIA ML Python for detailed info
            try:
                import nvidia_ml_py3 as nvml
                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                info = nvml.nvmlDeviceGetMemoryInfo(handle)
                device_name = nvml.nvmlDeviceGetName(handle)
                total_memory = info.total / (1024**3)
                
                if isinstance(device_name, bytes):
                    device_name = device_name.decode()
                
                print(f"📊 GPU Details: {device_name} ({total_memory:.1f}GB total)")
                
            except Exception as e:
                print(f"⚠️ Detailed GPU info unavailable: {e}")
                
                # Try PyTorch as fallback for device name
                try:
                    import torch
                    if torch.cuda.is_available():
                        device_name = torch.cuda.get_device_name(0)
                        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        print(f"📊 GPU Details (via PyTorch): {device_name} ({total_memory:.1f}GB)")
                    else:
                        print("⚠️ PyTorch CUDA not available, but ONNX CUDA is available")
                except Exception:
                    print("⚠️ PyTorch not available for GPU details")
            
            # Set GPU info with 4GB limit as requested
            gpu_info.update({
                'available': True,
                'memory_gb': 4.0,  # 4GB limit as requested
                'provider': 'CUDAExecutionProvider',
                'device_name': device_name,
                'total_memory': total_memory
            })
            
            print(f"✅ GPU configured: {device_name} (using 4GB limit)")
            return gpu_info
                
        except ImportError as e:
            print(f"❌ ONNX Runtime not available: {e}")
            print("💡 Install with: pip install onnxruntime-gpu")
        except Exception as e:
            print(f"❌ GPU detection failed: {e}")
        
        print("💻 No GPU detected, using CPU only")
        return gpu_info
    
    def _calculate_optimal_threads(self) -> int:
        """Calculate optimal thread count optimized for 6-core systems"""
        if self.cpu_cores == 6:
            # Optimized for 6-core systems: use 4 threads for processing
            # Reserve 2 cores for video pipeline and system operations
            print(f"🎯 6-core optimization: Using 4 threads (reserving 2 cores)")
            return 4
        elif self.cpu_cores >= 12:
            # High-end systems: use up to 8 threads
            return min(8, self.cpu_cores - 4)
        elif self.cpu_cores >= 8:
            # 8-core systems: use 6 threads
            return min(6, self.cpu_cores - 2)
        elif self.cpu_cores >= 4:
            # 4-core systems: use 3 threads
            return self.cpu_cores - 1
        else:
            # Low-core systems: conservative approach
            return max(1, self.cpu_cores // 2)
    
    def setup_environment(self):
        """Configure environment based on detected hardware with GPU optimization"""
        # Core threading optimization for 6-core systems
        os.environ['OMP_NUM_THREADS'] = str(self.optimal_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.optimal_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.optimal_threads)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(self.optimal_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.optimal_threads)
        
        # GPU memory optimization for 4GB limit
        if self.gpu_info['available']:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            # Limit GPU memory to 4GB
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            os.environ['TF_GPU_MEMORY_LIMIT'] = '4096'  # 4GB in MB
        
        # Memory optimization
        if self.memory_gb < 8:
            os.environ['OPENCV_VIDEOCAPTURE_CACHE_SIZE'] = '1'
            os.environ['OPENCV_DNN_OPENCL_ALLOW_ALL_DEVICES'] = '0'
        
        # ARM-specific optimizations (disable if we have good GPU)
        if self.is_arm and not self.gpu_info['available']:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU on ARM without GPU
            os.environ['OPENCV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2_USE_NEW_API'] = '1'
    
    def get_optimal_detection_size(self) -> tuple:
        """Get optimal detection size based on hardware capabilities"""
        if self.gpu_info['available']:
            # GPU available: can handle larger detection sizes
            if self.gpu_info['memory_gb'] >= 4:
                return (512, 512)  # Larger size for better accuracy
            else:
                return (416, 416)
        else:
            # CPU only: conservative sizes
            if self.memory_gb < 4:
                return (320, 320)
            elif self.memory_gb < 8:
                return (416, 416)
            else:
                return (512, 512)
    
    def get_processing_config(self) -> dict:
        """Get processing configuration optimized for 6-core system"""
        # Optimized configuration for 6-core systems
        config = {
            'max_faces': 8,  # Reasonable limit for 6-core system
            'frame_skip': 1,  # Process every frame for 30 FPS target
            'thread_pool_size': self.optimal_threads,
            'batch_size': 4,  # Optimized batch size for 6 cores
            'enable_caching': True,  # Enable caching for performance
            'gpu_enabled': self.gpu_info['available'],
            'gpu_memory_limit': self.gpu_info['memory_gb']
        }
        
        # Adjust based on memory
        if self.memory_gb >= 16:
            config['max_faces'] = 12
            config['batch_size'] = 6
        elif self.memory_gb >= 8:
            config['max_faces'] = 10
            config['batch_size'] = 5
        
        return config

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

class TemporalValidator:
    """Temporal consistency checker to reduce false positives across multiple frames"""
    
    def __init__(self, window_size: int = 5, consistency_threshold: float = 0.6):
        self.window_size = window_size  # Number of frames to consider
        self.consistency_threshold = consistency_threshold  # Required consistency ratio
        self.face_history = {}  # Track face detections over time
        self.frame_counter = 0
        
    def validate_face_recognition(self, face_embedding: np.ndarray, recognition_result: Tuple[str, float, Dict], 
                                face_bbox: List[int]) -> Tuple[str, float, Dict, bool]:
        """
        Validate face recognition using temporal consistency
        Returns: (name, confidence, metadata, is_temporally_consistent)
        """
        self.frame_counter += 1
        name, confidence, metadata = recognition_result
        
        # Create a face signature based on position and embedding similarity
        face_signature = self._create_face_signature(face_embedding, face_bbox)
        
        # Update face history
        if face_signature not in self.face_history:
            self.face_history[face_signature] = {
                'detections': [],
                'first_seen': self.frame_counter,
                'last_seen': self.frame_counter
            }
        
        # Add current detection
        self.face_history[face_signature]['detections'].append({
            'frame': self.frame_counter,
            'name': name,
            'confidence': confidence,
            'embedding': face_embedding.copy()
        })
        self.face_history[face_signature]['last_seen'] = self.frame_counter
        
        # Keep only recent detections within window
        cutoff_frame = self.frame_counter - self.window_size
        self.face_history[face_signature]['detections'] = [
            d for d in self.face_history[face_signature]['detections'] 
            if d['frame'] > cutoff_frame
        ]
        
        # Clean up old face signatures
        self._cleanup_old_signatures()
        
        # Check temporal consistency
        is_consistent = self._check_temporal_consistency(face_signature, name, confidence)
        
        # Adjust confidence based on temporal consistency
        if is_consistent:
            # Boost confidence for temporally consistent detections
            adjusted_confidence = min(1.0, confidence * 1.1)
            return name, adjusted_confidence, metadata, True
        else:
            # Reduce confidence for inconsistent detections
            adjusted_confidence = confidence * 0.8
            return name, adjusted_confidence, metadata, False
    
    def _create_face_signature(self, embedding: np.ndarray, bbox: List[int]) -> str:
        """Create a unique signature for a face based on position and embedding"""
        # Use bbox center and embedding hash for signature
        center_x = bbox[0] + bbox[2] // 2
        center_y = bbox[1] + bbox[3] // 2
        
        # Create a simple hash of the embedding
        embedding_hash = hash(tuple(embedding[:10]))  # Use first 10 elements for speed
        
        # Create signature with spatial tolerance (group nearby faces)
        spatial_bucket_x = center_x // 50  # 50-pixel tolerance
        spatial_bucket_y = center_y // 50
        
        return f"{spatial_bucket_x}_{spatial_bucket_y}_{embedding_hash}"
    
    def _check_temporal_consistency(self, face_signature: str, current_name: str, 
                                  current_confidence: float) -> bool:
        """Check if current recognition is consistent with recent history"""
        detections = self.face_history[face_signature]['detections']
        
        if len(detections) < 2:  # Need at least 2 detections for consistency check
            return True  # Assume consistent for new faces
        
        # Count how many recent detections match current result
        matching_detections = 0
        total_detections = len(detections)
        
        for detection in detections:
            if detection['name'] == current_name:
                matching_detections += 1
        
        # Calculate consistency ratio
        consistency_ratio = matching_detections / total_detections
        
        # Additional check: if confidence is very high, be more lenient
        if current_confidence > 0.8:
            consistency_threshold = self.consistency_threshold * 0.8
        else:
            consistency_threshold = self.consistency_threshold
        
        return consistency_ratio >= consistency_threshold
    
    def _cleanup_old_signatures(self):
        """Remove old face signatures to prevent memory growth"""
        cutoff_frame = self.frame_counter - (self.window_size * 2)
        signatures_to_remove = []
        
        for signature, data in self.face_history.items():
            if data['last_seen'] < cutoff_frame:
                signatures_to_remove.append(signature)
        
        for signature in signatures_to_remove:
            del self.face_history[signature]
    
    def get_statistics(self) -> Dict:
        """Get temporal validation statistics"""
        return {
            'tracked_faces': len(self.face_history),
            'current_frame': self.frame_counter,
            'window_size': self.window_size,
            'consistency_threshold': self.consistency_threshold
        }

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
        
        # Initialize InsightFace with hardware-optimized settings including GPU
        print(f"🔧 Initializing InsightFace model (Hardware: {system_optimizer.cpu_cores} cores, {system_optimizer.memory_gb:.1f}GB RAM)...")
        
        # Configure GPU for InsightFace
        if system_optimizer.gpu_info['available']:
            try:
                # Test GPU availability for InsightFace
                import onnxruntime as ort
                gpu_providers = ort.get_available_providers()
                print(f"🔧 Available providers: {gpu_providers}")
                
                if 'CUDAExecutionProvider' in gpu_providers:
                    print(f"🎮 Configuring InsightFace with GPU acceleration")
                    
                    # Add CUDA DLL paths to system PATH before initializing InsightFace
                    import site
                    import sys
                    
                    # Get site-packages directory
                    site_packages = None
                    for path in sys.path:
                        if path.endswith('site-packages'):
                            site_packages = path
                            break
                    
                    if not site_packages:
                        # Fallback method
                        site_packages = os.path.dirname(os.path.dirname(onnxruntime.__file__))
                    
                    if site_packages:
                        cuda_paths = [
                            # NVIDIA CUDA libraries - these work for both CUDA 11.x and 12.x
                            os.path.join(site_packages, 'nvidia', 'cublas', 'bin'),
                            os.path.join(site_packages, 'nvidia', 'cudnn', 'bin'), 
                            os.path.join(site_packages, 'nvidia', 'cuda_runtime', 'bin'),
                            os.path.join(site_packages, 'nvidia', 'cuda_cupti', 'bin'),
                            os.path.join(site_packages, 'nvidia', 'cufft', 'bin'),
                            os.path.join(site_packages, 'nvidia', 'cusolver', 'bin'),
                            os.path.join(site_packages, 'nvidia', 'cusparse', 'bin'),
                            os.path.join(site_packages, 'nvidia', 'curand', 'bin'),
                            os.path.join(site_packages, 'nvidia', 'nvjitlink', 'bin')
                        ]
                        
                        # Add DLL directories to DLL search path (Windows-specific)
                        if hasattr(os, 'add_dll_directory'):
                            for cuda_path in cuda_paths:
                                if os.path.exists(cuda_path):
                                    try:
                                        os.add_dll_directory(cuda_path)
                                        print(f"📁 Added DLL directory: {cuda_path}")
                                    except Exception as e:
                                        print(f"⚠️ Failed to add DLL directory {cuda_path}: {e}")
                        
                        # Also add to PATH as backup
                        current_path = os.environ.get('PATH', '')
                        for cuda_path in cuda_paths:
                            if os.path.exists(cuda_path) and cuda_path not in current_path:
                                os.environ['PATH'] = cuda_path + os.pathsep + current_path
                                current_path = os.environ['PATH']
                                print(f"📁 Added to PATH: {cuda_path}")
                    
                    # Set CUDA environment before initializing InsightFace
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                    
                    # Initialize FaceAnalysis with explicit GPU support
                    # InsightFace automatically uses GPU if available
                    self.face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                    
                    # Prepare with GPU context (ctx_id=0 for GPU)
                    self.face_app.prepare(ctx_id=0, det_size=self.detection_size)
                    
                    # Verify GPU usage
                    print(f"✅ InsightFace initialized with GPU acceleration")
                    print(f"🎮 GPU: {system_optimizer.gpu_info.get('device_name', 'Unknown')} (4GB limit)")
                    print(f"📏 Detection size: {self.detection_size}")
                    
                    # Check which providers are actually being used
                    if hasattr(self.face_app, 'models'):
                        gpu_models = 0
                        total_models = 0
                        for model_name, model in self.face_app.models.items():
                            total_models += 1
                            if hasattr(model, 'session') and hasattr(model.session, 'get_providers'):
                                providers = model.session.get_providers()
                                using_gpu = 'CUDAExecutionProvider' in providers
                                if using_gpu:
                                    gpu_models += 1
                                status = "🎮 GPU" if using_gpu else "💻 CPU"
                                print(f"   {model_name}: {status} - {providers[0] if providers else 'Unknown'}")
                        
                        if gpu_models > 0:
                            print(f"🎉 SUCCESS: {gpu_models}/{total_models} models using GPU!")
                        else:
                            print(f"⚠️ WARNING: 0/{total_models} models using GPU, falling back to CPU")
                            system_optimizer.gpu_info['available'] = False
                    
                else:
                    raise Exception("CUDA provider not available")
                    
            except Exception as e:
                print(f"⚠️ GPU initialization failed ({e}), falling back to CPU")
                system_optimizer.gpu_info['available'] = False
                
                # Fallback to CPU
                self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
                self.face_app.prepare(ctx_id=-1, det_size=self.detection_size)
                print(f"✅ InsightFace initialized with CPU fallback - Detection: {self.detection_size}, Threads: {system_optimizer.optimal_threads}")
        else:
            # CPU-only initialization
            self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=-1, det_size=self.detection_size)
            print(f"✅ InsightFace initialized with CPU - Detection: {self.detection_size}, Threads: {system_optimizer.optimal_threads}")
        
        print(f"🎯 Config: Max faces: {self.config['max_faces']}, Batch size: {self.config['batch_size']}")
        
        # Load authorized persons with caching optimization
        self.authorized_persons = {}
        self.recognition_threshold = float(os.getenv('RECOGNITION_THRESHOLD', '0.6'))
        
        # Performance monitoring with temporal validation
        self._processing_stats = {
            'total_processed': 0,
            'avg_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'temporal_consistent': 0,
            'temporal_inconsistent': 0
        }
        
        # Initialize temporal validator for false positive reduction
        self.temporal_validator = TemporalValidator(window_size=5, consistency_threshold=0.6)
        
        # Image counter for consistent numbering
        self.image_counter = 1
        
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
    
    def get_next_image_number(self) -> int:
        """Get the next image number and increment counter"""
        current_number = self.image_counter
        self.image_counter += 1
        return current_number
    
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
        """Advanced image preprocessing to reduce false positives while maintaining 0.6 threshold"""
        height, width = image.shape[:2]
        
        # Calculate optimal size based on memory constraints
        max_pixels = int(system_optimizer.memory_gb * 300000)  # ~300k pixels per GB
        current_pixels = height * width
        
        # Create a copy for processing
        processed_image = image.copy()
        
        # 1. Histogram Equalization for better contrast
        if len(processed_image.shape) == 3:
            # Convert to LAB color space for better histogram equalization
            lab = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            processed_image = cv2.merge([l, a, b])
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed_image = clahe.apply(processed_image)
        
        # 2. Noise Reduction with bilateral filter
        # Bilateral filter reduces noise while preserving edges (important for face features)
        processed_image = cv2.bilateralFilter(processed_image, 9, 75, 75)
        
        # 3. Sharpening to enhance facial features
        # Create a sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        # Apply sharpening with reduced intensity to avoid artifacts
        sharpened = cv2.filter2D(processed_image, -1, kernel)
        # Blend original and sharpened (30% sharpened, 70% original)
        processed_image = cv2.addWeighted(processed_image, 0.7, sharpened, 0.3, 0)
        
        # 4. Gamma correction for lighting normalization
        # This helps with varying lighting conditions
        gamma = 1.2  # Slightly increase brightness
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        processed_image = cv2.LUT(processed_image, table)
        
        # 5. Resize if needed (after all processing)
        if current_pixels > max_pixels:
            scale = np.sqrt(max_pixels / current_pixels)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Ensure minimum size for face detection
            new_width = max(320, new_width)
            new_height = max(240, new_height)
            
            # Use high-quality interpolation for better results
            processed_image = cv2.resize(processed_image, (new_width, new_height), 
                                       interpolation=cv2.INTER_CUBIC)
        
        return processed_image
    
    def _assess_face_quality(self, face_region: np.ndarray, face_bbox: np.ndarray) -> dict:
        """Assess face quality to reduce false positives"""
        quality_score = 1.0
        quality_factors = {}
        
        try:
            # 1. Blur detection using Laplacian variance
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Higher variance means less blur (better quality)
            if blur_score < 100:  # Very blurry
                quality_score *= 0.3
                quality_factors['blur'] = 'high'
            elif blur_score < 300:  # Moderately blurry
                quality_score *= 0.7
                quality_factors['blur'] = 'medium'
            else:  # Sharp
                quality_factors['blur'] = 'low'
            
            # 2. Brightness assessment
            mean_brightness = np.mean(gray)
            if mean_brightness < 50:  # Too dark
                quality_score *= 0.5
                quality_factors['brightness'] = 'too_dark'
            elif mean_brightness > 200:  # Too bright (overexposed)
                quality_score *= 0.6
                quality_factors['brightness'] = 'too_bright'
            else:
                quality_factors['brightness'] = 'good'
            
            # 3. Face size assessment
            bbox_area = (face_bbox[2] - face_bbox[0]) * (face_bbox[3] - face_bbox[1])
            if bbox_area < 1600:  # 40x40 pixels - too small
                quality_score *= 0.4
                quality_factors['size'] = 'too_small'
            elif bbox_area < 6400:  # 80x80 pixels - small but acceptable
                quality_score *= 0.8
                quality_factors['size'] = 'small'
            else:
                quality_factors['size'] = 'good'
            
            # 4. Contrast assessment
            contrast = gray.std()
            if contrast < 20:  # Low contrast
                quality_score *= 0.6
                quality_factors['contrast'] = 'low'
            else:
                quality_factors['contrast'] = 'good'
            
        except Exception as e:
            print(f"⚠️ Face quality assessment error: {e}")
            quality_score = 0.5  # Conservative default
            quality_factors['error'] = str(e)
        
        return {
            'score': quality_score,
            'factors': quality_factors,
            'meets_threshold': quality_score >= 0.6  # Quality threshold
        }
    
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
        """Process a single face with optimization and quality assessment"""
        face_id = f"face_{index+1}"
        
        # Extract face information
        bbox = face.bbox.astype(int)
        x, y, x2, y2 = bbox
        w, h = x2 - x, y2 - y
        
        # Extract face region for quality assessment
        face_region = image[max(0, y):min(image.shape[0], y2), 
                           max(0, x):min(image.shape[1], x2)]
        
        # Assess face quality to reduce false positives
        quality_assessment = self._assess_face_quality(face_region, bbox)
        
        # Get face embedding
        face_embedding = face.normed_embedding
        
        # Recognize face using optimized method
        initial_name, initial_confidence, metadata = self.recognize_face_from_embedding(face_embedding)
        
        # Apply temporal validation to reduce false positives
        name, confidence, metadata, is_temporally_consistent = self.temporal_validator.validate_face_recognition(
            face_embedding, (initial_name, initial_confidence, metadata), [x, y, w, h]
        )
        
        # Update temporal statistics
        if is_temporally_consistent:
            self._processing_stats['temporal_consistent'] += 1
        else:
            self._processing_stats['temporal_inconsistent'] += 1
        
        # Apply quality-based confidence adjustment (without changing the 0.6 threshold)
        original_confidence = confidence
        if not quality_assessment['meets_threshold']:
            # Lower confidence for poor quality faces
            confidence = confidence * quality_assessment['score']
        
        # Prepare face result
        face_result = {
            'face_id': face_id,
            'recognition': {
                'name': name,
                'confidence': confidence,
                'original_confidence': original_confidence,
                'initial_confidence': initial_confidence,
                'authorized': name != "Unknown" and confidence >= self.recognition_threshold,
                'quality_adjusted': confidence != original_confidence,
                'temporally_consistent': is_temporally_consistent
            },
            'metadata': metadata,
            'bbox': [x, y, w, h],
            'age': int(face.age) if hasattr(face, 'age') else 0,
            'gender': 'Male' if (hasattr(face, 'sex') and face.sex == 1) else 'Female',
            'detection_confidence': float(face.det_score),
            'quality_assessment': quality_assessment
        }
        
        # Handle unauthorized persons with async upload
        if not face_result['recognition']['authorized']:
            # Only store high-quality unauthorized faces
            if quality_assessment['meets_threshold']:
                # Resize to standard 192x192 for Azure storage
                face_region = cv2.resize(face_region, (192, 192), interpolation=cv2.INTER_LINEAR)
                
                # Add timestamp overlay to the image
                timestamp_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
                face_region_with_timestamp = add_timestamp_overlay(
                    face_region, 
                    timestamp_str=timestamp_str,
                    confidence=confidence,
                    person_id=face_id
                )
                
                timestamp = start_time.strftime("%Y%m%d_%H%M%S")
                image_no = self.get_next_image_number()
                filename = f"unauthorized_{image_no}_{timestamp}.jpg"
                
                # Async image upload
                future = self.executor.submit(self._upload_image_to_azure, face_region_with_timestamp, filename)
                # Store future for later retrieval if needed
                face_result['upload_future'] = future
            else:
                # Mark low-quality faces as not stored
                face_result['quality_rejected'] = True
        
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
                'optimal_threads': system_optimizer.optimal_threads,
                'gpu_available': system_optimizer.gpu_info['available'],
                'gpu_memory_gb': system_optimizer.gpu_info['memory_gb'],
                'gpu_provider': system_optimizer.gpu_info.get('provider', 'None')
            },
            'configuration': self.config
        }
    
    def verify_gpu_usage(self) -> Dict[str, Any]:
        """Verify if GPU is actually being used by InsightFace"""
        gpu_status = {
            'gpu_configured': system_optimizer.gpu_info['available'],
            'cuda_available': False,
            'insightface_providers': [],
            'memory_usage': None
        }
        
        try:
            # Check ONNX Runtime providers
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            gpu_status['cuda_available'] = 'CUDAExecutionProvider' in available_providers
            
            # Check InsightFace model providers
            if hasattr(self.face_app, 'models'):
                for model_name, model in self.face_app.models.items():
                    if hasattr(model, 'session') and hasattr(model.session, 'get_providers'):
                        gpu_status['insightface_providers'].append({
                            'model': model_name,
                            'providers': model.session.get_providers()
                        })
            
            # Try to get GPU memory usage
            try:
                import nvidia_ml_py3 as nvml
                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                info = nvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_status['memory_usage'] = {
                    'used_mb': info.used / (1024 * 1024),
                    'total_mb': info.total / (1024 * 1024),
                    'utilization_percent': (info.used / info.total) * 100
                }
            except Exception:
                gpu_status['memory_usage'] = "Unable to query GPU memory"
                
        except Exception as e:
            gpu_status['error'] = str(e)
        
        return gpu_status
    
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