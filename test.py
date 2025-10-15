#!/usr/bin/env python3
"""
Live Feed Test for Optimized Face Recognition System
Uses the existing optimized main.py system but processes live camera feed
Stores only unique unauthorized persons to avoid redundancy
Auto-refreshes authorized persons every 5 seconds
Integrated with audio alert system for unauthorized person detection
"""

import cv2
import numpy as np
import sys
import os
from datetime import datetime
import time
import json
import threading
import importlib.util
import os
from typing import Dict, List, Tuple

# Import the optimized system from main.py
from main import OptimizedFaceRecognitionSystem

# Import audio alert system
try:
    audio_alert_path = os.path.join(os.path.dirname(__file__), "audio-alert.py")
    spec = importlib.util.spec_from_file_location("audio_alert", audio_alert_path)
    audio_alert = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audio_alert)
    AudioAlertSystem = audio_alert.AudioAlertSystem
    AUDIO_ALERTS_AVAILABLE = True
    print("✅ Audio alert system loaded successfully")
except Exception as e:
    AUDIO_ALERTS_AVAILABLE = False
    print(f"⚠️ Audio alert system not available: {e}")
    AudioAlertSystem = None

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
        location_text = location if location else "0.000000, 0.000000"
        
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

class AuthorizedPersonsRefresher:
    """Automatically refresh authorized persons from Azure every 5 seconds"""
    
    def __init__(self, face_processor):
        self.face_processor = face_processor
        self.refresh_interval = 5.0  # 5 seconds
        self.running = False
        self.refresh_thread = None
        self.last_refresh_time = 0
        self.refresh_count = 0
        
    def start_refresh(self):
        """Start the automatic refresh in a background thread"""
        if not self.running:
            self.running = True
            self.refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
            self.refresh_thread.start()
            print(f"🔄 Started automatic authorized persons refresh (every {self.refresh_interval}s)")
    
    def stop_refresh(self):
        """Stop the automatic refresh"""
        self.running = False
        if self.refresh_thread:
            self.refresh_thread.join(timeout=1.0)
        print("🛑 Stopped automatic authorized persons refresh")
    
    def _refresh_loop(self):
        """Background thread loop for refreshing authorized persons"""
        while self.running:
            try:
                # Wait for the refresh interval
                time.sleep(self.refresh_interval)
                
                if self.running:  # Check again after sleep
                    start_time = time.time()
                    
                    # Get current count before refresh
                    old_count = self.face_processor.get_authorized_persons_count()
                    
                    # Refresh the authorized persons from Azure
                    success = self.face_processor._load_embeddings_from_azure()
                    
                    if success:
                        new_count = self.face_processor.get_authorized_persons_count()
                        refresh_time = time.time() - start_time
                        self.refresh_count += 1
                        self.last_refresh_time = time.time()
                        
                        # Show update info
                        if new_count != old_count:
                            print(f"🔄 Authorized persons updated: {old_count} → {new_count} persons ({refresh_time:.2f}s) [Refresh #{self.refresh_count}]")
                        else:
                            print(f"🔄 Authorized persons checked: {new_count} persons (no changes) ({refresh_time:.2f}s) [Refresh #{self.refresh_count}]")
                    else:
                        print(f"⚠️ Failed to refresh authorized persons [Refresh #{self.refresh_count}]")
                        
            except Exception as e:
                print(f"❌ Error in authorized persons refresh: {e}")
                time.sleep(1)  # Brief pause before retrying
    
    def get_status(self):
        """Get refresh status information"""
        time_since_last = time.time() - self.last_refresh_time if self.last_refresh_time > 0 else 0
        return {
            'running': self.running,
            'refresh_count': self.refresh_count,
            'last_refresh_ago': time_since_last,
            'interval': self.refresh_interval
        }

class UniqueUnauthorizedTracker:
    """Root-level optimized tracker for unique unauthorized persons"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.unauthorized_database = {}  # Store embeddings and metadata of unauthorized persons
        self.session_detections = {}     # Track detections in current session
        self.next_unique_id = 1
        self.image_counter = 1  # Global counter for image numbering
        
        # Root-level optimization: Use efficient data structures
        self._embedding_cache = {}  # LRU cache for embeddings
        self._similarity_cache = {}  # Cache similarity calculations
        self._max_cache_size = 1000  # Reasonable cache limit
        
    def _manage_cache(self):
        """Root-level optimization: Intelligent cache management"""
        if len(self._similarity_cache) > self._max_cache_size:
            # Remove oldest 20% of cache entries
            cache_items = list(self._similarity_cache.items())
            keep_size = int(self._max_cache_size * 0.8)
            self._similarity_cache = dict(cache_items[-keep_size:])
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings using root-level optimizations"""
        try:
            # Create cache key for similarity calculation
            emb1_hash = hash(embedding1.tobytes())
            emb2_hash = hash(embedding2.tobytes())
            cache_key = (min(emb1_hash, emb2_hash), max(emb1_hash, emb2_hash))
            
            # Check cache first (root-level optimization)
            if cache_key in self._similarity_cache:
                return self._similarity_cache[cache_key]
            
            # Use optimized numpy operations for better performance
            # Normalize embeddings efficiently
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                # Vectorized dot product for performance
                similarity = float(np.dot(embedding1, embedding2) / (norm1 * norm2))
            
            # Cache the result
            self._similarity_cache[cache_key] = similarity
            self._manage_cache()
            
            return similarity
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def is_unique_unauthorized_person(self, face_embedding: np.ndarray) -> Tuple[bool, str]:
        """
        Check if this is a unique unauthorized person using root-level optimizations
        Returns: (is_unique, unique_id)
        """
        # Root-level optimization: Use efficient numpy operations
        face_embedding = np.array(face_embedding, dtype=np.float64)  # Maintain precision
        
        # Batch similarity calculation for better performance
        if len(self.unauthorized_database) > 0:
            stored_embeddings = []
            unique_ids = []
            
            for unique_id, data in self.unauthorized_database.items():
                stored_embeddings.append(data['embedding'])
                unique_ids.append(unique_id)
            
            # Vectorized similarity calculation (root-level optimization)
            if stored_embeddings:
                similarities = []
                for stored_embedding in stored_embeddings:
                    similarity = self.calculate_similarity(face_embedding, stored_embedding)
                    similarities.append(similarity)
                
                # Find best match
                max_similarity_idx = np.argmax(similarities)
                max_similarity = similarities[max_similarity_idx]
                
                if max_similarity >= self.similarity_threshold:
                    # This is a known unauthorized person
                    matched_id = unique_ids[max_similarity_idx]
                    self.session_detections[matched_id] = self.session_detections.get(matched_id, 0) + 1
                    return False, matched_id
        
        # This is a new unique unauthorized person
        unique_id = f"unauthorized_{self.next_unique_id:04d}"
        
        # Store with full precision for accuracy
        self.unauthorized_database[unique_id] = {
            'embedding': face_embedding,
            'first_seen': datetime.now().isoformat(),
            'detection_count': 1,
            'stored_image': False
        }
        self.session_detections[unique_id] = 1
        self.next_unique_id += 1
        
        return True, unique_id
    
    def should_store_image(self, unique_id: str) -> bool:
        """Determine if we should store an image for this unique person"""
        if unique_id not in self.unauthorized_database:
            return False
        
        # Store image only if we haven't stored one yet
        if not self.unauthorized_database[unique_id]['stored_image']:
            self.unauthorized_database[unique_id]['stored_image'] = True
            return True
        
        return False
    
    def get_session_stats(self) -> Dict:
        """Get statistics for current session"""
        total_unique = len(self.unauthorized_database)
        total_detections = sum(self.session_detections.values())
        
        return {
            'unique_unauthorized_persons': total_unique,
            'total_detections_session': total_detections,
            'detection_breakdown': dict(self.session_detections),
            'database_size': len(self.unauthorized_database)
        }
    
    def get_next_image_number(self) -> int:
        """Get the next image number and increment counter"""
        current_number = self.image_counter
        self.image_counter += 1
        return current_number

class AudioAlertManager:
    """Manages audio alerts for unauthorized person detection with severity-based alerts"""
    
    def __init__(self):
        self.audio_system = None
        self.last_alert_time = {}  # Track cooldown periods for different alert types
        self.cooldown_periods = {
            'single_unauthorized': 5,      # 5 seconds between single person alerts
            'multiple_unauthorized': 3,    # 3 seconds between multiple person alerts  
            'continuous_detection': 15     # 15 seconds between continuous detection alerts
        }
        self.consecutive_detections = 0
        self.last_detection_time = 0
        
        # Initialize audio system if available
        if AUDIO_ALERTS_AVAILABLE and AudioAlertSystem:
            try:
                self.audio_system = AudioAlertSystem()
                print("🔊 Audio alert manager initialized successfully")
            except Exception as e:
                print(f"⚠️ Failed to initialize audio alert system: {e}")
                self.audio_system = None
        else:
            print("⚠️ Audio alerts disabled - AudioAlertSystem not available")
    
    def process_detection_results(self, results: dict, frame_count: int):
        """Process detection results and trigger appropriate audio alerts"""
        if not self.audio_system:
            return  # Skip if audio system not available
        
        try:
            current_time = time.time()
            unauthorized_count = results.get('unauthorized_count', 0)
            unique_unauthorized_count = results.get('unique_unauthorized_count', 0)
            
            # Update consecutive detection tracking
            if unauthorized_count > 0:
                if current_time - self.last_detection_time < 2.0:  # Within 2 seconds
                    self.consecutive_detections += 1
                else:
                    self.consecutive_detections = 1
                self.last_detection_time = current_time
            else:
                self.consecutive_detections = 0
            
            # Determine alert severity and trigger appropriate sound
            if unauthorized_count >= 3:
                # CRITICAL: Multiple unauthorized persons detected
                self._trigger_alert('multiple_unauthorized', 'critical', current_time,
                                  f"CRITICAL: {unauthorized_count} unauthorized persons detected!")
                                  
            elif self.consecutive_detections >= 5:
                # HIGH: Continuous unauthorized activity  
                self._trigger_alert('continuous_detection', 'high', current_time,
                                  f"HIGH: Continuous unauthorized activity detected ({self.consecutive_detections} consecutive frames)")
                                  
            elif unauthorized_count > 0:
                # NORMAL: Single unauthorized person
                self._trigger_alert('single_unauthorized', 'normal', current_time,
                                  f"NORMAL: {unauthorized_count} unauthorized person(s) detected")
            
        except Exception as e:
            print(f"⚠️ Error in audio alert processing: {e}")
    
    def _trigger_alert(self, alert_type: str, priority: str, current_time: float, message: str):
        """Trigger audio alert if cooldown period has passed"""
        try:
            # Check cooldown period
            last_alert = self.last_alert_time.get(alert_type, 0)
            cooldown = self.cooldown_periods.get(alert_type, 5)
            
            if current_time - last_alert >= cooldown:
                # Play the alert sound
                print(f"🚨 {message}")
                success = self.audio_system.play_alert_sound(priority)
                
                if success:
                    self.last_alert_time[alert_type] = current_time
                    print(f"   ✅ Audio alert triggered: {priority.upper()} priority")
                else:
                    print(f"   ❌ Failed to play audio alert")
            else:
                # Still in cooldown period
                remaining_cooldown = cooldown - (current_time - last_alert)
                print(f"   ⏳ Audio alert cooldown: {remaining_cooldown:.1f}s remaining for {alert_type}")
                
        except Exception as e:
            print(f"❌ Error triggering audio alert: {e}")
    
    def get_alert_stats(self):
        """Get audio alert statistics"""
        if not self.audio_system:
            return {'audio_system': 'disabled'}
            
        current_time = time.time()
        stats = {
            'audio_system': 'enabled',
            'consecutive_detections': self.consecutive_detections,
            'cooldowns': {}
        }
        
        for alert_type, last_time in self.last_alert_time.items():
            cooldown = self.cooldown_periods.get(alert_type, 5)
            time_since_last = current_time - last_time
            remaining = max(0, cooldown - time_since_last)
            stats['cooldowns'][alert_type] = {
                'last_alert_ago': time_since_last,
                'cooldown_remaining': remaining
            }
            
        return stats

def verify_gpu_setup(processor):
    """Verify GPU configuration and usage"""
    print("\n" + "="*60)
    print("🎮 GPU VERIFICATION")
    print("="*60)
    
    try:
        gpu_status = processor.verify_gpu_usage()
        
        print(f"🔧 GPU Configured: {'✅ YES' if gpu_status['gpu_configured'] else '❌ NO'}")
        print(f"🚀 CUDA Available: {'✅ YES' if gpu_status['cuda_available'] else '❌ NO'}")
        
        if gpu_status['insightface_providers']:
            print(f"\n📋 InsightFace Model Providers:")
            for model_info in gpu_status['insightface_providers']:
                providers = model_info['providers']
                model_name = model_info['model']
                using_gpu = 'CUDAExecutionProvider' in providers
                status = "✅ GPU" if using_gpu else "💻 CPU"
                print(f"   {model_name}: {status} ({', '.join(providers)})")
        
        if gpu_status['memory_usage'] and isinstance(gpu_status['memory_usage'], dict):
            usage = gpu_status['memory_usage']
            print(f"\n💾 GPU Memory Usage:")
            print(f"   Used: {usage['used_mb']:.1f} MB")
            print(f"   Total: {usage['total_mb']:.1f} MB")
            print(f"   Utilization: {usage['utilization_percent']:.1f}%")
        elif gpu_status['memory_usage']:
            print(f"\n💾 GPU Memory: {gpu_status['memory_usage']}")
        
        if 'error' in gpu_status:
            print(f"\n⚠️ Error: {gpu_status['error']}")
            
    except Exception as e:
        print(f"❌ GPU verification failed: {e}")
    
    print("="*60)

def main():
    """Run live feed face recognition with solid root-level optimizations"""
    
    try:
        # Initialize the optimized system (same as main.py)
        print("🚀 Initializing Root-Level Optimized Face Recognition System...")
        system = OptimizedFaceRecognitionSystem()
        
        # Initialize unique unauthorized tracker
        print("🎯 Initializing Root-Level Optimized Unauthorized Tracker...")
        unauthorized_tracker = UniqueUnauthorizedTracker(similarity_threshold=0.8)
        
        # Initialize audio alert manager for unauthorized person detection
        print("🔊 Initializing Audio Alert Manager...")
        audio_alert_manager = AudioAlertManager()
        
        # Access the face processor for live feed processing
        processor = system.face_processor
        
        # Verify GPU setup
        verify_gpu_setup(processor)
        
        # Initialize automatic authorized persons refresher
        print("🔄 Initializing Authorized Persons Auto-Refresher...")
        auth_refresher = AuthorizedPersonsRefresher(processor)
        auth_refresher.start_refresh()
        
        # Get system capabilities dynamically (root-level optimization)
        import multiprocessing
        import platform
        import os
        
        available_cores = multiprocessing.cpu_count()
        
        # Detect Jetson hardware specifically
        is_jetson = False
        jetson_model = "Unknown"
        
        try:
            # Check for Jetson-specific files/hardware
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'r') as f:
                    model_info = f.read().strip()
                    if 'jetson' in model_info.lower() or 'nvidia' in model_info.lower():
                        is_jetson = True
                        jetson_model = model_info
            
            # Also check architecture
            arch = platform.machine()
            if arch in ['aarch64', 'arm64'] and available_cores <= 8:
                is_jetson = True
                jetson_model = f"ARM64 Device ({available_cores} cores)"
                
        except Exception:
            # Fallback detection for Jetson-like systems
            arch = platform.machine()
            if arch in ['aarch64', 'arm64'] and available_cores <= 8:
                is_jetson = True
                jetson_model = f"ARM64 Device ({available_cores} cores)"
        
        print(f"🔧 System Capabilities: {available_cores} CPU cores available")
        if is_jetson:
            print(f"🚀 JETSON DETECTED: {jetson_model}")
            print("⚡ Applying aggressive Jetson optimizations for 30+ FPS target")
        else:
            print("⚡ Using adaptive processing based on system resources")
        
        # Initialize webcam with adaptive settings
        print("📷 Initializing camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Unable to access camera")
            return
        
        # Adaptive camera settings based on system capabilities
        if is_jetson:  # Jetson-optimized for 30+ FPS
            # Aggressive settings for Jetson 30 FPS target
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)   # Optimized for YOLO/detection models
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)  # 4:3 aspect ratio, smaller resolution
            cap.set(cv2.CAP_PROP_FPS, 30)           # Target 30 FPS
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))  # MJPEG for performance
            print("🎯 JETSON 30+ FPS mode: 416x320 @ 30fps (MJPEG)")
        elif available_cores >= 8:  # High-performance system
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            print("🎯 High-performance mode: 1280x720 @ 30fps")
        elif available_cores >= 4:  # Medium-performance system
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            cap.set(cv2.CAP_PROP_FPS, 25)
            print("🎯 Medium-performance mode: 960x540 @ 25fps")
        else:  # Low-performance system
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)
            print("🎯 Low-performance mode: 640x480 @ 15fps")
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for low latency
        
        # Jetson-specific additional optimizations
        if is_jetson:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto exposure for consistent timing
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)         # Disable autofocus for speed
            print("� Jetson camera optimizations: Fixed exposure, no autofocus")
        
        print("�📹 Starting adaptive live face recognition...")
        if is_jetson:
            print("🚀 JETSON TARGET: 30+ FPS with aggressive optimizations")
        print("🔧 Press 'q' to quit, 's' for stats, 'u' for unique stats, 'r' for refresh stats, 'a' for audio stats")
        
        frame_count = 0
        fps_start_time = time.time()
        total_processing_time = 0.0
        processed_frames = 0
        
        # Adaptive frame processing based on system capabilities
        # Initialize adaptive FPS controller for 30 FPS target
        fps_controller = AdaptiveFPSController(target_fps=30.0, window_size=30)
        print(f"🎯 Adaptive FPS Controller initialized: Target 30 FPS")
        
        # Jetson-specific performance tracking (legacy support)
        jetson_fps_window = []
        jetson_target_fps = 30.0
        last_fps_check = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Determine if frame should be processed using adaptive controller
            should_process = fps_controller.should_process_frame(frame_count)
            
            if should_process:
                processed_frames += 1
                
                # Start timing
                start_time = time.time()
                
                # Use adaptive processing optimized for 30 FPS
                results = process_frame_adaptive_30fps(processor, frame, frame_count, 
                                                     unauthorized_tracker, fps_controller)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                
                # Update FPS controller with timing information
                fps_controller.update_frame_timing(processing_time)
                
                # Log detections to Azure (every processed frame)
                if results.get('faces'):
                    log_data = {
                        'timestamp': datetime.now().isoformat(),
                        'source': 'live_camera',
                        'frame_number': frame_count,
                        'faces': []
                    }
                    
                    for face in results['faces']:
                        face_log = {
                            'name': face['recognition']['name'],
                            'confidence': face['recognition']['confidence'],
                            'authorized': face['recognition']['authorized']
                        }
                        log_data['faces'].append(face_log)
                    
                    # Log to Azure asynchronously
                    processor._log_prediction_to_azure_optimized(log_data)
                
                # Process audio alerts based on detection results
                audio_alert_manager.process_detection_results(results, frame_count)
                
                # Draw results on frame with adaptive quality
                draw_results_adaptive_30fps(frame, results, processing_time, fps_controller)
                
                # Show performance info periodically
                if processed_frames % 15 == 0:  # Every 15 processed frames
                    fps_stats = fps_controller.get_stats()
                    fps = frame_count / (current_time - fps_start_time)
                    avg_processing_time = total_processing_time / processed_frames
                    
                    stats = unauthorized_tracker.get_session_stats()
                    
                    print(f"🎯 30FPS Frame {frame_count}: {results['total_faces']} faces, "
                          f"{results['authorized_count']} auth, "
                          f"{stats['unique_unauthorized_persons']} unique, "
                          f"Current FPS: {fps_stats.get('current_fps', 0):.1f}, "
                          f"Quality: {fps_stats.get('quality_level', 'medium')}, "
                          f"Skip: {fps_stats.get('frame_skip', 1)}, "
                          f"Process: {avg_processing_time:.3f}s")
            else:
                # Frame skipped - still draw previous results if available
                if 'results' in locals():
                    draw_results_adaptive_30fps(frame, results, 0.0, fps_controller, skipped=True)
            
            # Display frame
            cv2.imshow('Adaptive Face Recognition System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_system_stats(processor, frame_count, total_processing_time, processed_frames)
            elif key == ord('u'):
                show_unauthorized_stats(unauthorized_tracker)
            elif key == ord('r'):
                show_refresh_stats(auth_refresher)
            elif key == ord('a'):
                show_audio_alert_stats(audio_alert_manager)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Stop the auto-refresher
        auth_refresher.stop_refresh()
        
        # Show final statistics
        print("\n📈 Root-Level Optimized System Final Statistics:")
        show_final_statistics(frame_count, fps_start_time, total_processing_time, 
                            processed_frames, processor, unauthorized_tracker, available_cores)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup resources
        if 'auth_refresher' in locals():
            auth_refresher.stop_refresh()
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        if 'system' in locals():
            system.face_processor.cleanup()
        # Final memory cleanup
        import gc
        gc.collect()

def draw_results_adaptive_30fps(frame: np.ndarray, results: dict, processing_time: float, 
                               fps_controller, skipped: bool = False):
    """Draw results with adaptive UI optimized for 30 FPS performance"""
    
    height, width = frame.shape[:2]
    fps_stats = fps_controller.get_stats()
    quality_level = fps_stats.get('quality_level', 'medium')
    current_fps = fps_stats.get('current_fps', 0)
    
    # Adaptive header size based on performance
    header_height = 90 if quality_level == 'high' else 75 if quality_level == 'medium' else 60
    
    # Header background with performance color coding
    if current_fps >= 28:
        header_color = (0, 40, 0)  # Dark green for excellent performance
    elif current_fps >= 24:
        header_color = (0, 30, 30)  # Dark yellow for good performance
    else:
        header_color = (0, 0, 40)  # Dark red for poor performance
    
    cv2.rectangle(frame, (0, 0), (width, header_height), header_color, -1)
    
    # Performance status indicator
    fps_status = "🟢" if current_fps >= 28 else "🟡" if current_fps >= 24 else "🔴"
    
    # Main info line with FPS emphasis
    if not skipped:
        header_text = f"{fps_status} 30FPS: {current_fps:.1f} | Faces:{results.get('total_faces', 0)} Auth:{results.get('authorized_count', 0)}"
    else:
        header_text = f"{fps_status} 30FPS: {current_fps:.1f} | FRAME SKIPPED"
    
    font_size = 0.7 if quality_level == 'high' else 0.6 if quality_level == 'medium' else 0.5
    cv2.putText(frame, header_text, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
    
    # Performance and quality info
    if not skipped:
        perf_text = f"Quality: {quality_level.upper()} | Skip: {fps_stats.get('frame_skip', 1)} | Process: {processing_time:.3f}s"
        cv2.putText(frame, perf_text, (5, 44), cv2.FONT_HERSHEY_SIMPLEX, font_size - 0.1, (0, 255, 255), 1)
        
        # Unique tracking info
        if 'unique_unauthorized_count' in results:
            unique_text = f"Unique Unauthorized: {results['unique_unauthorized_count']} | Stored: {results.get('unique_unauthorized_stored', 0)}"
            cv2.putText(frame, unique_text, (5, 66), cv2.FONT_HERSHEY_SIMPLEX, font_size - 0.2, (255, 255, 255), 1)
    
    # System status
    status_text = f"☁️ GPU+Azure | 6-Core Optimized | Quality: {quality_level}"
    cv2.putText(frame, status_text, (5, header_height - 5), cv2.FONT_HERSHEY_SIMPLEX, font_size - 0.2, (255, 255, 255), 1)
    
    # Draw face boxes with adaptive detail (only if not skipped)
    if not skipped and 'faces' in results:
        for face in results['faces']:
            bbox = face.get('bbox', [])
            if len(bbox) != 4:
                continue
            
            x, y, w, h = bbox
            authorized = face['recognition']['authorized']
            is_new_unique = face.get('is_new_unique', False)
            confidence = face['recognition']['confidence']
            temporally_consistent = face['recognition'].get('temporally_consistent', True)
            
            # Adaptive color coding with temporal consistency
            if authorized:
                color = (0, 255, 0) if temporally_consistent else (0, 200, 100)  # Green variants
                label = f"✓ {face['recognition']['name']}"
            else:
                if is_new_unique:
                    color = (0, 0, 255) if temporally_consistent else (0, 0, 200)  # Red variants
                    label = f"✗ NEW"
                else:
                    color = (0, 165, 255) if temporally_consistent else (0, 120, 200)  # Orange variants
                    label = f"✗ {face.get('unique_id', 'UNK')}"
            
            # Adaptive box thickness and detail based on quality level
            if quality_level == 'high':
                thickness = 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                # Show full details
                label_with_conf = f"{label} ({confidence:.2f})"
                cv2.putText(frame, label_with_conf, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Quality indicators
                if 'quality_assessment' in face:
                    quality_score = face['quality_assessment']['score']
                    quality_indicator = f"Q:{quality_score:.2f}"
                    cv2.putText(frame, quality_indicator, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                
            elif quality_level == 'medium':
                thickness = 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                # Show basic details
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
            else:  # low quality
                thickness = 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                # Minimal labels
                simple_label = "✓" if authorized else "✗"
                cv2.putText(frame, simple_label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Temporal consistency indicator (subtle)
            if not temporally_consistent and quality_level in ['high', 'medium']:
                cv2.circle(frame, (x + w - 10, y + 10), 3, (0, 255, 255), -1)  # Yellow dot for inconsistency

def process_frame_jetson_30fps(processor, frame, frame_count, unauthorized_tracker, batch_process):
    """Ultra-optimized frame processing for Jetson 30+ FPS target"""
    
    # Aggressive frame preprocessing for speed
    height, width = frame.shape[:2]
    
    # Resize to even smaller detection size for speed
    if width > 320:
        scale = 320 / width
        new_width = 320
        new_height = int(height * scale)
        processed_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)  # Fastest interpolation
    else:
        processed_frame = frame
    
    # Get face detections with strict limits for 30 FPS
    faces = processor.face_app.get(processed_frame)
    
    # Ultra-aggressive face limiting for 30+ FPS
    max_faces = 2  # Only 2 faces max for speed
    if len(faces) > max_faces:
        faces = sorted(faces, key=lambda x: x.det_score, reverse=True)[:max_faces]
    
    results = {
        'total_faces': len(faces),
        'authorized_count': 0,
        'unauthorized_count': 0,
        'unique_unauthorized_count': 0,
        'faces': [],
        'unique_unauthorized_stored': 0,
        'jetson_30fps_mode': True
    }
    
    # Process faces with batching for efficiency
    for i, face in enumerate(faces):
        face_id = f"face_{i+1}"
        
        # Extract face information with scaling back
        bbox = face.bbox.astype(int)
        if processed_frame is not frame:
            scale_factor = width / processed_frame.shape[1]
            bbox = (bbox * scale_factor).astype(int)
        
        x, y, x2, y2 = bbox
        w, h = x2 - x, y2 - y
        
        # Get face embedding with reduced precision for speed
        face_embedding = face.normed_embedding.astype(np.float32)  # Use float32 for speed
        
        # Fast recognition with simplified threshold
        name, confidence, metadata = processor.recognize_face_from_embedding(face_embedding)
        authorized = name != "Unknown" and confidence >= 0.65  # Slightly higher threshold for speed
        
        face_result = {
            'face_id': face_id,
            'recognition': {
                'name': name,
                'confidence': confidence,
                'authorized': authorized
            },
            'bbox': [x, y, w, h],
            'unique_id': None,
            'is_new_unique': False
        }
        
        if authorized:
            results['authorized_count'] += 1
        else:
            results['unauthorized_count'] += 1
            
            # Simplified uniqueness check for speed (every N frames)
            if frame_count % batch_process == 0:  # Only check uniqueness in batches
                is_unique, unique_id = unauthorized_tracker.is_unique_unauthorized_person(face_embedding)
                face_result['unique_id'] = unique_id
                face_result['is_new_unique'] = is_unique
                
                if is_unique:
                    results['unique_unauthorized_count'] += 1
                    
                    # Store image with minimal processing
                    if unauthorized_tracker.should_store_image(unique_id):
                        face_region = frame[max(0, y):min(frame.shape[0], y2), 
                                          max(0, x):min(frame.shape[1], x2)]
                        
                        if face_region.size > 0:
                            # Resize to standard 192x192 for Azure storage
                            face_region = cv2.resize(face_region, (192, 192), interpolation=cv2.INTER_LINEAR)
                            
                            # Add timestamp overlay to the image
                            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            face_region_with_timestamp = add_timestamp_overlay(
                                face_region, 
                                timestamp_str=timestamp_str,
                                confidence=confidence,
                                person_id=unique_id
                            )
                            
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            image_no = unauthorized_tracker.get_next_image_number()
                            filename = f"unauthorized_{image_no}_{timestamp}.jpg"
                            
                            # Async upload for speed (fire and forget)
                            try:
                                image_url = processor._upload_image_to_azure(face_region_with_timestamp, filename)
                                if image_url:
                                    results['unique_unauthorized_stored'] += 1
                                    print(f"⚡ Jetson30FPS stored: {unique_id} with timestamp (192x192)")
                            except Exception:
                                pass  # Silent fail for speed
        
        results['faces'].append(face_result)
    
    return results

def draw_results_jetson_minimal(frame: np.ndarray, results: dict, processing_time: float):
    """Ultra-minimal UI for Jetson 30+ FPS performance"""
    
    height, width = frame.shape[:2]
    
    # Minimal header (absolute minimum for 30+ FPS)
    cv2.rectangle(frame, (0, 0), (width, 30), (0, 0, 0), -1)
    
    # Essential info only - ultra-compact
    header_text = f"F:{results['total_faces']} A:{results['authorized_count']} U:{results.get('unique_unauthorized_count', 0)} T:{processing_time:.2f}s"
    cv2.putText(frame, header_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Minimal face indicators - just colored dots for speed
    for face in results.get('faces', []):
        bbox = face.get('bbox', [])
        if len(bbox) != 4:
            continue
        
        x, y, w, h = bbox
        authorized = face['recognition']['authorized']
        is_new_unique = face.get('is_new_unique', False)
        
        # Ultra-minimal indicators - just center dots
        center_x, center_y = x + w // 2, y + h // 2
        
        if authorized:
            color = (0, 255, 0)  # Green dot
        else:
            color = (0, 0, 255) if is_new_unique else (0, 165, 255)  # Red/Orange dot
        
        # Just a small filled circle - fastest drawing
        cv2.circle(frame, (center_x, center_y), 8, color, -1)
        
        # Optional: tiny rectangle for face area (very thin)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

def process_frame_optimized(processor, frame, frame_count, unauthorized_tracker, available_cores):
    """Process frame with adaptive root-level optimizations based on system capabilities"""
    
    # Adaptive frame resizing based on system performance
    height, width = frame.shape[:2]
    
    if available_cores >= 8:  # High-performance system
        # No resizing for high-performance systems
        processed_frame = frame
        max_faces = 16  # Allow more faces
    elif available_cores >= 4:  # Medium-performance system
        # Moderate resizing
        if width > 960:
            scale = 960 / width
            new_width = 960
            new_height = int(height * scale)
            processed_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            processed_frame = frame
        max_faces = 10
    else:  # Low-performance system
        # More aggressive resizing
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            processed_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            processed_frame = frame
        max_faces = 6
    
    # Get face detections with adaptive limits
    faces = processor.face_app.get(processed_frame)
    
    # Prioritize faces by detection confidence (root-level optimization)
    if len(faces) > max_faces:
        faces = sorted(faces, key=lambda x: x.det_score, reverse=True)[:max_faces]
    
    results = {
        'total_faces': len(faces),
        'authorized_count': 0,
        'unauthorized_count': 0,
        'unique_unauthorized_count': 0,
        'faces': [],
        'unique_unauthorized_stored': 0,
        'system_performance': 'high' if available_cores >= 8 else 'medium' if available_cores >= 4 else 'low'
    }
    
    for i, face in enumerate(faces):
        face_id = f"face_{i+1}"
        
        # Extract face information with proper scaling
        bbox = face.bbox.astype(int)
        if processed_frame is not frame:  # If we resized, scale back bbox
            scale_factor = width / processed_frame.shape[1]
            bbox = (bbox * scale_factor).astype(int)
        
        x, y, x2, y2 = bbox
        w, h = x2 - x, y2 - y
        
        # Get face embedding (maintain full precision)
        face_embedding = face.normed_embedding
        
        # Recognize face using the processor's method
        name, confidence, metadata = processor.recognize_face_from_embedding(face_embedding)
        authorized = name != "Unknown" and confidence >= processor.recognition_threshold
        
        face_result = {
            'face_id': face_id,
            'recognition': {
                'name': name,
                'confidence': confidence,
                'authorized': authorized
            },
            'bbox': [x, y, w, h],
            'unique_id': None,
            'is_new_unique': False
        }
        
        if authorized:
            results['authorized_count'] += 1
        else:
            results['unauthorized_count'] += 1
            
            # Check uniqueness using optimized tracker
            is_unique, unique_id = unauthorized_tracker.is_unique_unauthorized_person(face_embedding)
            face_result['unique_id'] = unique_id
            face_result['is_new_unique'] = is_unique
            
            if is_unique:
                results['unique_unauthorized_count'] += 1
                
                # Store image only for new unique unauthorized persons
                if unauthorized_tracker.should_store_image(unique_id):
                    # Extract face region from original frame
                    face_region = frame[max(0, y):min(frame.shape[0], y2), 
                                      max(0, x):min(frame.shape[1], x2)]
                    
                    # Standard resize to 192x192 for Azure storage
                    if face_region.size > 0:
                        # Always resize to 192x192 for consistent Azure storage
                        face_region = cv2.resize(face_region, (192, 192), interpolation=cv2.INTER_LINEAR)
                        
                        # Add timestamp overlay
                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        face_region_with_timestamp = add_timestamp_overlay(
                            face_region,
                            timestamp_str=timestamp_str
                        )
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_no = unauthorized_tracker.get_next_image_number()
                        filename = f"unauthorized_{image_no}_{timestamp}.jpg"
                        
                        # Upload to Azure with error handling
                        try:
                            image_url = processor._upload_image_to_azure(face_region_with_timestamp, filename)
                            if image_url:
                                face_result['stored_image_url'] = image_url
                                results['unique_unauthorized_stored'] += 1
                                print(f"📤 Stored unique person: {unique_id} with timestamp (192x192)")
                        except Exception as e:
                            print(f"⚠️  Upload failed (continuing): {e}")
        
        results['faces'].append(face_result)
    
    return results
    """Process frame with adaptive root-level optimizations based on system capabilities"""
    
    # Adaptive frame resizing based on system performance
    height, width = frame.shape[:2]
    
    if available_cores >= 8:  # High-performance system
        # No resizing for high-performance systems
        processed_frame = frame
        max_faces = 16  # Allow more faces
    elif available_cores >= 4:  # Medium-performance system
        # Moderate resizing
        if width > 960:
            scale = 960 / width
            new_width = 960
            new_height = int(height * scale)
            processed_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            processed_frame = frame
        max_faces = 10
    else:  # Low-performance system
        # More aggressive resizing
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            processed_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            processed_frame = frame
        max_faces = 6
    
    # Get face detections with adaptive limits
    faces = processor.face_app.get(processed_frame)
    
    # Prioritize faces by detection confidence (root-level optimization)
    if len(faces) > max_faces:
        faces = sorted(faces, key=lambda x: x.det_score, reverse=True)[:max_faces]
    
    results = {
        'total_faces': len(faces),
        'authorized_count': 0,
        'unauthorized_count': 0,
        'unique_unauthorized_count': 0,
        'faces': [],
        'unique_unauthorized_stored': 0,
        'system_performance': 'high' if available_cores >= 8 else 'medium' if available_cores >= 4 else 'low'
    }
    
    for i, face in enumerate(faces):
        face_id = f"face_{i+1}"
        
        # Extract face information with proper scaling
        bbox = face.bbox.astype(int)
        if processed_frame is not frame:  # If we resized, scale back bbox
            scale_factor = width / processed_frame.shape[1]
            bbox = (bbox * scale_factor).astype(int)
        
        x, y, x2, y2 = bbox
        w, h = x2 - x, y2 - y
        
        # Get face embedding (maintain full precision)
        face_embedding = face.normed_embedding
        
        # Recognize face using the processor's method
        name, confidence, metadata = processor.recognize_face_from_embedding(face_embedding)
        authorized = name != "Unknown" and confidence >= processor.recognition_threshold
        
        face_result = {
            'face_id': face_id,
            'recognition': {
                'name': name,
                'confidence': confidence,
                'authorized': authorized
            },
            'bbox': [x, y, w, h],
            'unique_id': None,
            'is_new_unique': False
        }
        
        if authorized:
            results['authorized_count'] += 1
        else:
            results['unauthorized_count'] += 1
            
            # Check uniqueness using optimized tracker
            is_unique, unique_id = unauthorized_tracker.is_unique_unauthorized_person(face_embedding)
            face_result['unique_id'] = unique_id
            face_result['is_new_unique'] = is_unique
            
            if is_unique:
                results['unique_unauthorized_count'] += 1
                
                # Store image only for new unique unauthorized persons
                if unauthorized_tracker.should_store_image(unique_id):
                    # Extract face region from original frame
                    face_region = frame[max(0, y):min(frame.shape[0], y2), 
                                      max(0, x):min(frame.shape[1], x2)]
                    
                    # Standard resize to 192x192 for Azure storage
                    if face_region.size > 0:
                        # Always resize to 192x192 for consistent Azure storage
                        face_region = cv2.resize(face_region, (192, 192), interpolation=cv2.INTER_LINEAR)
                        
                        # Add timestamp overlay to the image
                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        face_region_with_timestamp = add_timestamp_overlay(
                            face_region, 
                            timestamp_str=timestamp_str,
                            confidence=confidence,
                            person_id=unique_id
                        )
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_no = unauthorized_tracker.get_next_image_number()
                        filename = f"unauthorized_{image_no}_{timestamp}.jpg"
                        
                        # Upload to Azure with error handling
                        try:
                            image_url = processor._upload_image_to_azure(face_region_with_timestamp, filename)
                            if image_url:
                                face_result['stored_image_url'] = image_url
                                results['unique_unauthorized_stored'] += 1
                                print(f"📤 Stored unique person: {unique_id} with timestamp (192x192)")
                        except Exception as e:
                            print(f"⚠️  Upload failed (continuing): {e}")
        
        results['faces'].append(face_result)
    
    return results

def draw_results_optimized(frame: np.ndarray, results: dict, processing_time: float, unauthorized_tracker):
    """Draw results with adaptive UI based on system performance"""
    
    height, width = frame.shape[:2]
    performance_level = results.get('system_performance', 'medium')
    
    # Adaptive header size based on performance
    header_height = 80 if performance_level == 'high' else 65 if performance_level == 'medium' else 50
    
    # Header background
    cv2.rectangle(frame, (0, 0), (width, header_height), (0, 0, 0), -1)
    
    # Main info line
    header_text = f"Faces:{results['total_faces']} Auth:{results['authorized_count']} Unique:{results.get('unique_unauthorized_count', 0)}"
    font_size = 0.7 if performance_level == 'high' else 0.6 if performance_level == 'medium' else 0.5
    cv2.putText(frame, header_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
    
    # Performance info
    perf_text = f"Performance: {performance_level.upper()} | Time: {processing_time:.3f}s"
    cv2.putText(frame, perf_text, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size - 0.1, (0, 255, 255), 1)
    
    # System status
    cv2.putText(frame, "☁️ Cloud Processing", (5, header_height - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size - 0.2, (255, 255, 255), 1)
    
    # Draw face boxes with adaptive detail
    for face in results.get('faces', []):
        bbox = face.get('bbox', [])
        if len(bbox) != 4:
            continue
        
        x, y, w, h = bbox
        authorized = face['recognition']['authorized']
        is_new_unique = face.get('is_new_unique', False)
        confidence = face['recognition']['confidence']
        
        # Adaptive color coding
        if authorized:
            color = (0, 255, 0)  # Green
            label = f"✓ {face['recognition']['name']}"
        else:
            if is_new_unique:
                color = (0, 0, 255)  # Red for new unique
                label = f"✗ NEW"
            else:
                color = (0, 165, 255)  # Orange for known unauthorized
                label = f"✗ {face['unique_id']}"
        
        # Adaptive box thickness based on performance
        thickness = 2 if performance_level == 'high' else 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # Adaptive label detail
        if performance_level == 'high':
            # Show full details on high-performance systems
            label_with_conf = f"{label} ({confidence:.2f})"
            cv2.putText(frame, label_with_conf, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        elif performance_level == 'medium':
            # Show basic details
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        else:
            # Minimal labels for low-performance systems
            simple_label = "✓" if authorized else "✗"
            cv2.putText(frame, simple_label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

class AdaptiveFPSController:
    """Adaptive FPS controller to maintain consistent 30 FPS performance"""
    
    def __init__(self, target_fps: float = 30.0, window_size: int = 30):
        self.target_fps = target_fps
        self.window_size = window_size
        self.frame_times = []
        self.processing_times = []
        self.last_frame_time = time.time()
        self.frame_skip = 1
        self.quality_level = 'high'  # high, medium, low
        
        # Adaptive thresholds
        self.fps_tolerance = 0.9  # 90% of target FPS
        self.fps_excellent = 0.98  # 98% of target FPS
        
    def update_frame_timing(self, processing_time: float):
        """Update timing statistics and adjust processing strategy"""
        current_time = time.time()
        frame_interval = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Update circular buffers
        self.frame_times.append(frame_interval)
        self.processing_times.append(processing_time)
        
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
            self.processing_times.pop(0)
        
        # Calculate current performance
        if len(self.frame_times) >= 5:  # Need some samples
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            
            # Adaptive strategy adjustment
            self._adjust_strategy(current_fps, avg_processing_time)
    
    def _adjust_strategy(self, current_fps: float, avg_processing_time: float):
        """Adjust processing strategy based on performance"""
        target_threshold = self.target_fps * self.fps_tolerance
        excellent_threshold = self.target_fps * self.fps_excellent
        
        if current_fps >= excellent_threshold:
            # Excellent performance - can increase quality
            if self.quality_level == 'medium':
                self.quality_level = 'high'
                self.frame_skip = 1
            elif self.quality_level == 'low':
                self.quality_level = 'medium'
                self.frame_skip = max(1, self.frame_skip - 1)
        
        elif current_fps >= target_threshold:
            # Good performance - maintain current settings
            pass
        
        elif current_fps >= target_threshold * 0.8:
            # Below target but not critical - minor adjustments
            if self.quality_level == 'high':
                self.quality_level = 'medium'
            elif self.frame_skip < 2:
                self.frame_skip = 2
        
        else:
            # Critical performance - aggressive optimization
            if self.quality_level == 'high':
                self.quality_level = 'medium'
                self.frame_skip = 2
            elif self.quality_level == 'medium':
                self.quality_level = 'low'
                self.frame_skip = min(3, self.frame_skip + 1)
            else:
                self.frame_skip = min(4, self.frame_skip + 1)
    
    def should_process_frame(self, frame_count: int) -> bool:
        """Determine if frame should be processed based on adaptive skip strategy"""
        return frame_count % self.frame_skip == 0
    
    def get_processing_config(self) -> dict:
        """Get current processing configuration"""
        configs = {
            'high': {
                'max_faces': 8,
                'detection_size': (512, 512),
                'enable_quality_assessment': True,
                'enable_temporal_validation': True,
                'preprocessing_level': 'full'
            },
            'medium': {
                'max_faces': 6,
                'detection_size': (416, 416),
                'enable_quality_assessment': True,
                'enable_temporal_validation': True,
                'preprocessing_level': 'moderate'
            },
            'low': {
                'max_faces': 4,
                'detection_size': (320, 320),
                'enable_quality_assessment': False,
                'enable_temporal_validation': False,
                'preprocessing_level': 'minimal'
            }
        }
        return configs.get(self.quality_level, configs['medium'])
    
    def get_stats(self) -> dict:
        """Get current performance statistics"""
        if not self.frame_times:
            return {'status': 'initializing'}
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        
        return {
            'current_fps': current_fps,
            'target_fps': self.target_fps,
            'frame_skip': self.frame_skip,
            'quality_level': self.quality_level,
            'avg_processing_time': avg_processing_time,
            'performance_ratio': current_fps / self.target_fps if self.target_fps > 0 else 0
        }

def process_frame_adaptive_30fps(processor, frame, frame_count, unauthorized_tracker, fps_controller):
    """Adaptive frame processing optimized for 30 FPS with quality scaling"""
    
    # Get current processing configuration
    config = fps_controller.get_processing_config()
    
    # Adaptive preprocessing based on quality level
    if config['preprocessing_level'] == 'full':
        # Full preprocessing with all enhancements
        processed_frame = processor._preprocess_image(frame)
    elif config['preprocessing_level'] == 'moderate':
        # Moderate preprocessing - skip some enhancements
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            processed_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            processed_frame = frame
    else:
        # Minimal preprocessing - just resize if needed
        height, width = frame.shape[:2]
        if width > 416:
            scale = 416 / width
            new_width = 416
            new_height = int(height * scale)
            processed_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        else:
            processed_frame = frame
    
    # Get face detections with adaptive limits
    faces = processor.face_app.get(processed_frame)
    
    # Limit faces based on quality level
    max_faces = config['max_faces']
    if len(faces) > max_faces:
        faces = sorted(faces, key=lambda x: x.det_score, reverse=True)[:max_faces]
    
    results = {
        'total_faces': len(faces),
        'authorized_count': 0,
        'unauthorized_count': 0,
        'unique_unauthorized_count': 0,
        'faces': [],
        'unique_unauthorized_stored': 0,
        'processing_config': config,
        'fps_stats': fps_controller.get_stats()
    }
    
    # Process faces with adaptive quality
    for i, face in enumerate(faces):
        face_id = f"face_{i+1}"
        
        # Extract face information with scaling back if needed
        bbox = face.bbox.astype(int)
        if processed_frame is not frame:
            scale_factor = frame.shape[1] / processed_frame.shape[1]
            bbox = (bbox * scale_factor).astype(int)
        
        x, y, x2, y2 = bbox
        w, h = x2 - x, y2 - y
        
        # Get face embedding
        face_embedding = face.normed_embedding
        
        # Apply quality assessment based on config
        if config['enable_quality_assessment']:
            face_region = frame[max(0, y):min(frame.shape[0], y2), 
                              max(0, x):min(frame.shape[1], x2)]
            quality_assessment = processor._assess_face_quality(face_region, bbox)
        else:
            quality_assessment = {'score': 1.0, 'meets_threshold': True, 'factors': {}}
        
        # Face recognition with optional temporal validation
        if config['enable_temporal_validation']:
            initial_name, initial_confidence, metadata = processor.recognize_face_from_embedding(face_embedding)
            name, confidence, metadata, is_temporally_consistent = processor.temporal_validator.validate_face_recognition(
                face_embedding, (initial_name, initial_confidence, metadata), [x, y, w, h]
            )
        else:
            name, confidence, metadata = processor.recognize_face_from_embedding(face_embedding)
            is_temporally_consistent = True
            initial_confidence = confidence
        
        # Apply quality-based confidence adjustment
        original_confidence = confidence
        if config['enable_quality_assessment'] and not quality_assessment['meets_threshold']:
            confidence = confidence * quality_assessment['score']
        
        authorized = name != "Unknown" and confidence >= processor.recognition_threshold
        
        face_result = {
            'face_id': face_id,
            'recognition': {
                'name': name,
                'confidence': confidence,
                'original_confidence': original_confidence,
                'authorized': authorized,
                'temporally_consistent': is_temporally_consistent
            },
            'bbox': [x, y, w, h],
            'quality_assessment': quality_assessment,
            'unique_id': None,
            'is_new_unique': False
        }
        
        if authorized:
            results['authorized_count'] += 1
        else:
            results['unauthorized_count'] += 1
            
            # Unique tracking for unauthorized persons
            is_unique, unique_id = unauthorized_tracker.is_unique_unauthorized_person(face_embedding)
            face_result['unique_id'] = unique_id
            face_result['is_new_unique'] = is_unique
            
            if is_unique:
                results['unique_unauthorized_count'] += 1
                
                # Store image only for high-quality unique unauthorized persons
                if (unauthorized_tracker.should_store_image(unique_id) and 
                    quality_assessment['meets_threshold']):
                    
                    face_region = frame[max(0, y):min(frame.shape[0], y2), 
                                      max(0, x):min(frame.shape[1], x2)]
                    
                    if face_region.size > 0:
                        # Resize to standard 192x192 for Azure storage
                        face_region = cv2.resize(face_region, (192, 192), interpolation=cv2.INTER_LINEAR)
                        
                        # Add timestamp overlay
                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        face_region_with_timestamp = add_timestamp_overlay(
                            face_region, 
                            timestamp_str=timestamp_str,
                            confidence=confidence,
                            person_id=unique_id
                        )
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_no = unauthorized_tracker.get_next_image_number()
                        filename = f"unauthorized_{image_no}_{timestamp}.jpg"
                        
                        # Async upload for performance
                        try:
                            image_url = processor._upload_image_to_azure(face_region_with_timestamp, filename)
                            if image_url:
                                results['unique_unauthorized_stored'] += 1
                                print(f"🎯 30FPS stored: {unique_id} with timestamp (192x192)")
                        except Exception:
                            pass  # Silent fail for performance
        
        results['faces'].append(face_result)
    
    return results
    """Process frame and track unique unauthorized persons"""
    
    # Get face detections from the processor
    faces = processor.face_app.get(frame)
    
    # Limit number of faces if needed
    if len(faces) > processor.config['max_faces']:
        faces = sorted(faces, key=lambda x: x.det_score, reverse=True)[:processor.config['max_faces']]
    
    results = {
        'total_faces': len(faces),
        'authorized_count': 0,
        'unauthorized_count': 0,
        'unique_unauthorized_count': 0,
        'faces': [],
        'unique_unauthorized_stored': 0
    }
    
    for i, face in enumerate(faces):
        face_id = f"face_{i+1}"
        
        # Extract face information
        bbox = face.bbox.astype(int)
        x, y, x2, y2 = bbox
        w, h = x2 - x, y2 - y
        
        # Get face embedding
        face_embedding = face.normed_embedding
        
        # Recognize face using the processor's method
        name, confidence, metadata = processor.recognize_face_from_embedding(face_embedding)
        authorized = name != "Unknown" and confidence >= processor.recognition_threshold
        
        face_result = {
            'face_id': face_id,
            'recognition': {
                'name': name,
                'confidence': confidence,
                'authorized': authorized
            },
            'metadata': metadata,
            'bbox': [x, y, w, h],
            'age': int(face.age) if hasattr(face, 'age') else 0,
            'gender': 'Male' if (hasattr(face, 'sex') and face.sex == 1) else 'Female',
            'detection_confidence': float(face.det_score),
            'unique_id': None,
            'is_new_unique': False
        }
        
        if authorized:
            results['authorized_count'] += 1
        else:
            results['unauthorized_count'] += 1
            
            # Check if this is a unique unauthorized person
            is_unique, unique_id = unauthorized_tracker.is_unique_unauthorized_person(face_embedding)
            face_result['unique_id'] = unique_id
            face_result['is_new_unique'] = is_unique
            
            if is_unique:
                results['unique_unauthorized_count'] += 1
                
                # Store image only for new unique unauthorized persons
                if unauthorized_tracker.should_store_image(unique_id):
                    face_region = frame[y:y2, x:x2]
                    
                    # Resize to standard 192x192 for Azure storage
                    face_region = cv2.resize(face_region, (192, 192), interpolation=cv2.INTER_LINEAR)
                    
                    # Add timestamp overlay to the image
                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    face_region_with_timestamp = add_timestamp_overlay(
                        face_region, 
                        timestamp_str=timestamp_str,
                        confidence=confidence,
                        person_id=unique_id
                    )
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_no = unauthorized_tracker.get_next_image_number()
                    filename = f"unauthorized_{image_no}_{timestamp}.jpg"
                    
                    # Upload to Azure (async)
                    try:
                        image_url = processor._upload_image_to_azure(face_region_with_timestamp, filename)
                        if image_url:
                            face_result['stored_image_url'] = image_url
                            results['unique_unauthorized_stored'] += 1
                            print(f"📤 Stored unique unauthorized person: {unique_id} with timestamp (192x192)")
                    except Exception as e:
                        print(f"❌ Error storing unique unauthorized image: {e}")
        
        results['faces'].append(face_result)
    
    return results

def show_refresh_stats(auth_refresher):
    """Display authorized persons auto-refresh statistics"""
    print("\n" + "="*60)
    print("🔄 AUTHORIZED PERSONS AUTO-REFRESH STATISTICS")
    print("="*60)
    
    status = auth_refresher.get_status()
    
    print(f"📊 Refresh Status:")
    print(f"   Auto-Refresh: {'🟢 RUNNING' if status['running'] else '🔴 STOPPED'}")
    print(f"   Refresh Interval: {status['interval']}s")
    print(f"   Total Refreshes: {status['refresh_count']}")
    
    if status['last_refresh_ago'] > 0:
        print(f"   Last Refresh: {status['last_refresh_ago']:.1f}s ago")
        next_refresh = max(0, status['interval'] - status['last_refresh_ago'])
        print(f"   Next Refresh: {next_refresh:.1f}s")
    else:
        print(f"   Last Refresh: Not yet performed")
        print(f"   Next Refresh: Soon...")
    
    # Get current authorized persons count
    current_count = auth_refresher.face_processor.get_authorized_persons_count()
    print(f"\n👥 Current Authorized Persons: {current_count}")
    
    print("="*60)

def show_unauthorized_stats(unauthorized_tracker):
    """Display detailed unauthorized person statistics"""
    print("\n" + "="*60)
    print("👤 UNIQUE UNAUTHORIZED PERSONS STATISTICS")
    print("="*60)
    
    stats = unauthorized_tracker.get_session_stats()
    
    print(f"📊 Session Summary:")
    print(f"   Unique Unauthorized Persons: {stats['unique_unauthorized_persons']}")
    print(f"   Total Detections This Session: {stats['total_detections_session']}")
    print(f"   Database Size: {stats['database_size']}")
    
    if stats['detection_breakdown']:
        print(f"\n🔍 Detection Breakdown:")
        for unique_id, count in stats['detection_breakdown'].items():
            person_data = unauthorized_tracker.unauthorized_database.get(unique_id, {})
            first_seen = person_data.get('first_seen', 'Unknown')
            stored_image = person_data.get('stored_image', False)
            
            print(f"   {unique_id}: {count} detections")
            print(f"      First seen: {first_seen}")
            print(f"      Image stored: {'✅' if stored_image else '❌'}")
    else:
        print(f"\n✅ No unauthorized persons detected this session")
    
    print("="*60)

def show_audio_alert_stats(audio_alert_manager):
    """Display audio alert system statistics"""
    print("\n" + "="*60)
    print("🔊 AUDIO ALERT SYSTEM STATISTICS")
    print("="*60)
    
    stats = audio_alert_manager.get_alert_stats()
    
    if stats.get('audio_system') == 'disabled':
        print("❌ Audio Alert System: DISABLED")
        print("   Audio alerts are not available")
    else:
        print("✅ Audio Alert System: ENABLED")
        print(f"📊 Detection Tracking:")
        print(f"   Consecutive Detections: {stats.get('consecutive_detections', 0)}")
        
        if 'cooldowns' in stats and stats['cooldowns']:
            print(f"\n⏱️ Alert Cooldowns:")
            for alert_type, cooldown_info in stats['cooldowns'].items():
                last_ago = cooldown_info.get('last_alert_ago', 0)
                remaining = cooldown_info.get('cooldown_remaining', 0)
                
                alert_name = alert_type.replace('_', ' ').title()
                print(f"   {alert_name}:")
                print(f"      Last alert: {last_ago:.1f}s ago")
                if remaining > 0:
                    print(f"      Cooldown remaining: {remaining:.1f}s")
                else:
                    print(f"      Status: Ready for next alert")
        else:
            print(f"\n⏱️ No alerts triggered yet this session")
        
        print(f"\n🔊 Alert Priorities:")
        print(f"   NORMAL: Single unauthorized person (2 beeps)")
        print(f"   HIGH: Continuous unauthorized activity (3 beeps)")
        print(f"   CRITICAL: Multiple unauthorized persons (5 beeps)")
        
        print(f"\n🎛️ Cooldown Settings:")
        print(f"   Single unauthorized: 5 seconds")
        print(f"   Multiple unauthorized: 3 seconds")
        print(f"   Continuous detection: 15 seconds")
    
    print("="*60)

def show_system_stats(processor, frame_count: int, total_processing_time: float, processed_frames: int):
    """Show comprehensive system statistics with root-level optimization info"""
    print("\n" + "="*60)
    print("📊 ROOT-LEVEL OPTIMIZED SYSTEM STATISTICS")
    print("="*60)
    
    # Hardware info from system optimizer
    from main import system_optimizer
    print(f"🖥️  Hardware Configuration:")
    print(f"   CPU Cores: {system_optimizer.cpu_cores}")
    print(f"   Memory: {system_optimizer.memory_gb:.1f}GB")
    print(f"   Architecture: {'ARM64' if system_optimizer.is_arm else 'x86_64'}")
    print(f"   Optimal Threads: {system_optimizer.optimal_threads}")
    
    # Root-level optimizations applied
    print(f"\n⚡ Root-Level Optimizations:")
    print(f"   Dynamic Threading: ✅ Enabled")
    print(f"   Vectorized Computation: ✅ Enabled")
    print(f"   Intelligent Caching: ✅ Enabled")
    print(f"   Adaptive Processing: ✅ Based on hardware capabilities")
    
    # Performance metrics
    perf_stats = processor.get_performance_stats()
    print(f"\n📈 Performance Metrics:")
    print(f"   Total Frames Captured: {frame_count}")
    print(f"   Frames Processed: {processed_frames}")
    print(f"   Processing Efficiency: {(processed_frames/frame_count)*100:.1f}%")
    print(f"   Total Processing Time: {total_processing_time:.2f}s")
    print(f"   Avg Processing Time: {total_processing_time/processed_frames:.3f}s per frame")
    print(f"   Cache Hits: {perf_stats['cache_hits']}")
    print(f"   Cache Misses: {perf_stats['cache_misses']}")
    print(f"   Cache Efficiency: {(perf_stats['cache_hits']/(perf_stats['cache_hits']+perf_stats['cache_misses']))*100:.1f}%")
    
    # Recognition capabilities
    authorized_count = processor.get_authorized_persons_count()
    print(f"\n👥 Recognition Database:")
    print(f"   Authorized Persons: {authorized_count}")
    
    print("="*60)

def show_final_statistics(frame_count, fps_start_time, total_processing_time, 
                         processed_frames, processor, unauthorized_tracker, available_cores):
    """Show comprehensive final statistics for root-level optimized system"""
    current_time = time.time()
    total_time = current_time - fps_start_time
    avg_fps = frame_count / total_time
    avg_processing_time = total_processing_time / processed_frames if processed_frames > 0 else 0
    
    print(f"🎥 Total frames captured: {frame_count}")
    print(f"⚡ Frames processed: {processed_frames}")
    print(f"📊 Average FPS: {avg_fps:.1f}")
    print(f"🕐 Average processing time: {avg_processing_time:.3f}s")
    
    # System performance classification
    if available_cores >= 8:
        performance_class = "HIGH-PERFORMANCE"
    elif available_cores >= 4:
        performance_class = "MEDIUM-PERFORMANCE" 
    else:
        performance_class = "RESOURCE-CONSTRAINED"
    
    print(f"🖥️  System Classification: {performance_class} ({available_cores} cores)")
    
    # Recognition performance stats
    perf_stats = processor.get_performance_stats()
    print(f"📋 Recognition performance: {perf_stats['total_processed']} total processed, "
          f"Cache hits: {perf_stats['cache_hits']}, Cache misses: {perf_stats['cache_misses']}")
    
    # Unique unauthorized statistics
    unauthorized_stats = unauthorized_tracker.get_session_stats()
    total_detections = unauthorized_stats['total_detections_session']
    unique_persons = unauthorized_stats['unique_unauthorized_persons']
    
    print(f"\n👤 Unique Unauthorized Person Summary:")
    print(f"   Total unique persons identified: {unique_persons}")
    print(f"   Total unauthorized detections: {total_detections}")
    
    if unique_persons > 0 and total_detections > 0:
        efficiency = ((total_detections - unique_persons) / total_detections) * 100
        print(f"   Images stored: {unique_persons} (instead of {total_detections})")
        print(f"   Storage efficiency: {efficiency:.1f}% reduction in redundant storage")
    
    # Root-level optimization summary
    print(f"🚀 Root-level optimizations active: Adaptive processing, intelligent caching, vectorized computation")
    if unauthorized_stats['unique_unauthorized_persons'] > 0:
        # Calculate storage efficiency
        stored_images = sum(1 for data in unauthorized_tracker.unauthorized_database.values() 
                          if data.get('stored_image', False))
        storage_efficiency = (1 - (stored_images / unauthorized_stats['total_detections_session'])) * 100
        
        print(f"   Images stored: {stored_images} (instead of {unauthorized_stats['total_detections_session']})")
        print(f"   Storage efficiency: {storage_efficiency:.1f}% reduction in redundant storage")
    
    print(f"🚀 System optimizations: {processor.config}")

def draw_results_on_frame(frame: np.ndarray, results: dict, processing_time: float, unauthorized_tracker):
    """Draw recognition results on the live frame with unique tracking info"""
    
    # Draw header info
    height, width = frame.shape[:2]
    
    # Background for header
    cv2.rectangle(frame, (0, 0), (width, 100), (0, 0, 0), -1)
    
    # Header text with unique tracking
    header_text = f"Faces: {results['total_faces']} | Authorized: {results['authorized_count']}"
    cv2.putText(frame, header_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Unique unauthorized info
    unique_text = f"Unauthorized: {results['unauthorized_count']} | Unique: {results.get('unique_unauthorized_count', 0)}"
    cv2.putText(frame, unique_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Performance and storage info
    perf_text = f"Processing: {processing_time:.2f}s | Stored: {results.get('unique_unauthorized_stored', 0)} unique images"
    cv2.putText(frame, perf_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Cloud status
    cv2.putText(frame, "☁️ AZURE + UNIQUE TRACKING", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw face detection results
    for face in results.get('faces', []):
        bbox = face.get('bbox', [])
        if len(bbox) != 4:
            continue
        
        x, y, w, h = bbox
        name = face['recognition']['name']
        confidence = face['recognition']['confidence']
        authorized = face['recognition']['authorized']
        unique_id = face.get('unique_id')
        is_new_unique = face.get('is_new_unique', False)
        
        # Choose colors based on authorization and uniqueness
        if authorized:
            color = (0, 255, 0)  # Green for authorized
        else:
            if is_new_unique:
                color = (0, 0, 255)  # Red for new unique unauthorized
            else:
                color = (0, 165, 255)  # Orange for known unauthorized
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Prepare label
        if authorized:
            label = f"✅ {name}"
            confidence_text = f"{confidence:.2f}"
        else:
            if is_new_unique:
                label = f"🆕 NEW UNAUTHORIZED"
                confidence_text = f"{unique_id}"
            else:
                label = f"👤 KNOWN UNAUTHORIZED"
                confidence_text = f"{unique_id}"
        
        # Draw name label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x, y - 35), (x + label_size[0] + 10, y), color, -1)
        
        # Draw name label
        cv2.putText(frame, label, (x + 5, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw unique ID or confidence
        cv2.putText(frame, confidence_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add special indicator for new unique unauthorized persons
        if is_new_unique:
            cv2.putText(frame, "� STORED", (x + w - 80, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

def show_system_stats(processor, frame_count: int, total_processing_time: float):
    """Display detailed system statistics"""
    print("\n" + "="*60)
    print("📊 SYSTEM STATISTICS")
    print("="*60)
    
    # Hardware info
    from main import system_optimizer
    print(f"🖥️  Hardware:")
    print(f"   CPU Cores: {system_optimizer.cpu_cores}")
    print(f"   Memory: {system_optimizer.memory_gb:.1f}GB")
    print(f"   Architecture: {'ARM64' if system_optimizer.is_arm else 'x86_64'}")
    print(f"   Optimal Threads: {system_optimizer.optimal_threads}")
    
    # Configuration
    print(f"\n⚙️  Configuration:")
    config = processor.config
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Performance stats
    perf_stats = processor.get_performance_stats()
    print(f"\n📈 Performance:")
    print(f"   Total Processed: {perf_stats['total_processed']}")
    print(f"   Avg Processing Time: {perf_stats['avg_processing_time']:.3f}s")
    print(f"   Cache Hits: {perf_stats['cache_hits']}")
    print(f"   Cache Misses: {perf_stats['cache_misses']}")
    
    # Live feed stats
    print(f"\n📹 Live Feed:")
    print(f"   Frames Captured: {frame_count}")
    print(f"   Total Processing Time: {total_processing_time:.2f}s")
    
    # Authorized persons
    authorized_count = processor.get_authorized_persons_count()
    print(f"\n👥 Authorized Persons: {authorized_count}")
    authorized_names = processor.get_authorized_persons_list()
    for i, name in enumerate(authorized_names, 1):
        print(f"   {i}. {name}")
    
    print("="*60)

if __name__ == "__main__":
    main()
