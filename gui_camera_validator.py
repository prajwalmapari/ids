#!/usr/bin/env python3
"""
GUI-Based Live Performance Validation System with Camera Selection
================================================================

A comprehensive tkinter-based GUI application for validating face recognition 
system performance with user-selectable camera input and real-time metrics.

Features:
- Camera selection dropdown (based on available cameras)
- Live video feed display
- Real-time performance metrics
- Face detection and authorization tracking
- System resource monitoring
- Interactive start/stop/reset controls
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import threading
import time
import psutil
import numpy as np
from PIL import Image, ImageTk
import sys
import os
from collections import deque

# Import face recognition system
try:
    from main import OptimizedFaceProcessor
except ImportError:
    print("Warning: Could not import OptimizedFaceProcessor from main.py")
    OptimizedFaceProcessor = None

class CameraPerformanceValidator:
    """GUI-based camera performance validation system."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Performance Validator")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1e1e1e')
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.processing_times = deque(maxlen=30)
        self.start_time = None
        self.frame_count = 0
        self.total_faces = 0
        self.total_authorized = 0
        self.total_unauthorized = 0
        self.error_count = 0
        self.critical_alerts = 0
        
        # FPS and Performance Management
        self.target_fps = 15  # Target FPS for optimal performance
        self.max_fps = 30     # Maximum FPS
        self.min_fps = 5      # Minimum FPS
        self.frame_skip_count = 0
        self.last_frame_time = 0
        self.frame_interval = 1.0 / self.target_fps
        self.adaptive_quality = True
        self.current_quality_scale = 1.0
        
        # Processing optimization
        self.face_detection_interval = 2  # Process faces every N frames
        self.face_detection_counter = 0
        self.last_face_results = []
        self.processing_queue_size = 3
        self.skip_heavy_processing = False
        self.faces_processed_count = 0  # Track actual face processing operations
        
        # Camera and processing
        self.camera = None
        self.selected_camera = tk.StringVar()
        self.available_cameras = []
        self.is_running = False
        self.face_processor = None
        
        # Threading
        self.camera_thread = None
        self.face_processing_thread = None
        self.metrics_thread = None
        self.stop_threads = False
        
        # Frame sharing between threads
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.processing_frame = None
        self.processing_lock = threading.Lock()
        
        # UI Components
        self.video_label = None
        self.metrics_vars = {}
        
        # Initialize
        self.detect_cameras()
        self.setup_face_processor()
        self.setup_ui()
        self.start_metrics_update()
        
    def detect_cameras(self):
        """Detect available cameras."""
        print("🔍 Detecting available cameras...")
        self.available_cameras = []
        
        for i in range(5):  # Check first 5 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.available_cameras.append(i)
                    print(f"✅ Camera {i}: Available")
                cap.release()
            else:
                print(f"❌ Camera {i}: Not available")
        
        if not self.available_cameras:
            self.available_cameras = [0]  # Fallback
            print("⚠️ No cameras detected, using default camera 0")
        
        print(f"📹 Found {len(self.available_cameras)} available cameras: {self.available_cameras}")
        
    def setup_face_processor(self):
        """Initialize face recognition processor."""
        try:
            if OptimizedFaceProcessor:
                self.face_processor = OptimizedFaceProcessor()
                print("✅ Face recognition processor initialized")
            else:
                print("⚠️ Face recognition processor not available")
        except Exception as e:
            print(f"❌ Error initializing face processor: {e}")
            self.face_processor = None
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main title
        title_frame = tk.Frame(self.root, bg='#1e1e1e')
        title_frame.pack(fill='x', padx=20, pady=10)
        
        title_label = tk.Label(
            title_frame,
            text="🎥 Face Recognition Performance Validator",
            font=('Arial', 18, 'bold'),
            fg='#ffffff',
            bg='#1e1e1e'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Select camera and monitor real-time system performance",
            font=('Arial', 10),
            fg='#cccccc',
            bg='#1e1e1e'
        )
        subtitle_label.pack()
        
        # Control panel
        self.setup_control_panel()
        
        # Main content area
        content_frame = tk.Frame(self.root, bg='#1e1e1e')
        content_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Video panel (left side)
        self.setup_video_panel(content_frame)
        
        # Metrics panel (right side)
        self.setup_metrics_panel(content_frame)
        
    def setup_control_panel(self):
        """Setup camera selection and control buttons."""
        control_frame = tk.Frame(self.root, bg='#2d2d2d', relief='ridge', bd=2)
        control_frame.pack(fill='x', padx=20, pady=(0, 10))
        
        # Camera selection
        camera_frame = tk.Frame(control_frame, bg='#2d2d2d')
        camera_frame.pack(side='left', fill='y', padx=10, pady=10)
        
        tk.Label(
            camera_frame,
            text="Camera:",
            font=('Arial', 10, 'bold'),
            fg='#ffffff',
            bg='#2d2d2d'
        ).pack(side='left', padx=(0, 5))
        
        camera_options = [f"Camera {i}" for i in self.available_cameras]
        if camera_options:
            self.selected_camera.set(camera_options[0])
        
        camera_combo = ttk.Combobox(
            camera_frame,
            textvariable=self.selected_camera,
            values=camera_options,
            state='readonly',
            width=12
        )
        camera_combo.pack(side='left', padx=(0, 10))
        camera_combo.bind('<<ComboboxSelected>>', self.on_camera_changed)
        
        # Performance settings
        perf_frame = tk.Frame(control_frame, bg='#2d2d2d')
        perf_frame.pack(side='left', fill='y', padx=10, pady=10)
        
        tk.Label(
            perf_frame,
            text="Target FPS:",
            font=('Arial', 10, 'bold'),
            fg='#ffffff',
            bg='#2d2d2d'
        ).pack(side='left', padx=(0, 5))
        
        self.fps_var = tk.StringVar(value=str(self.target_fps))
        fps_combo = ttk.Combobox(
            perf_frame,
            textvariable=self.fps_var,
            values=['5', '10', '15', '20', '25', '30'],
            state='readonly',
            width=5
        )
        fps_combo.pack(side='left', padx=(0, 10))
        fps_combo.bind('<<ComboboxSelected>>', self.on_fps_changed)
        
        # Quality toggle
        self.quality_var = tk.BooleanVar(value=self.adaptive_quality)
        quality_check = tk.Checkbutton(
            perf_frame,
            text="Adaptive Quality",
            variable=self.quality_var,
            command=self.on_quality_changed,
            font=('Arial', 9),
            fg='#ffffff',
            bg='#2d2d2d',
            selectcolor='#1e1e1e'
        )
        quality_check.pack(side='left', padx=(0, 10))
        
        # Control buttons
        button_frame = tk.Frame(control_frame, bg='#2d2d2d')
        button_frame.pack(side='right', padx=10, pady=10)
        
        self.start_btn = tk.Button(
            button_frame,
            text="🚀 Start Validation",
            font=('Arial', 10, 'bold'),
            bg='#28a745',
            fg='white',
            command=self.start_validation,
            width=15
        )
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = tk.Button(
            button_frame,
            text="🛑 Stop",
            font=('Arial', 10, 'bold'),
            bg='#dc3545',
            fg='white',
            command=self.stop_validation,
            width=10,
            state='disabled'
        )
        self.stop_btn.pack(side='left', padx=5)
        
        self.reset_btn = tk.Button(
            button_frame,
            text="🔄 Reset Stats",
            font=('Arial', 10, 'bold'),
            bg='#007bff',
            fg='white',
            command=self.reset_stats,
            width=12
        )
        self.reset_btn.pack(side='left', padx=5)
    
    def setup_video_panel(self, parent):
        """Setup video display panel."""
        video_frame = tk.Frame(parent, bg='#2d2d2d', relief='ridge', bd=2)
        video_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Video title
        tk.Label(
            video_frame,
            text="📹 Live Video Feed",
            font=('Arial', 14, 'bold'),
            fg='#ffffff',
            bg='#2d2d2d'
        ).pack(pady=10)
        
        # Video display
        self.video_label = tk.Label(
            video_frame,
            bg='#000000',
            text="Select camera and click 'Start Validation'\nto begin live feed",
            fg='#ffffff',
            font=('Arial', 12),
            width=50,
            height=20
        )
        self.video_label.pack(expand=True, fill='both', padx=20, pady=(0, 20))
    
    def setup_metrics_panel(self, parent):
        """Setup metrics display panel."""
        metrics_frame = tk.Frame(parent, bg='#2d2d2d', relief='ridge', bd=2, width=300)
        metrics_frame.pack(side='right', fill='y')
        metrics_frame.pack_propagate(False)
        
        # Metrics title
        tk.Label(
            metrics_frame,
            text="📊 Performance Metrics",
            font=('Arial', 14, 'bold'),
            fg='#ffffff',
            bg='#2d2d2d'
        ).pack(pady=10)
        
        # Create metrics sections
        self.create_metrics_section(metrics_frame, "⚡ Performance", [
            ("FPS", "fps", "0.0"),
            ("Target FPS", "target_fps", str(self.target_fps)),
            ("Processing Time", "processing_time", "0.000s"),
            ("CPU Usage", "cpu_usage", "0.0%"),
            ("Memory Usage", "memory_usage", "0.0%"),
            ("Quality Scale", "quality_scale", "100%")
        ])
        
        self.create_metrics_section(metrics_frame, "👥 Face Detection", [
            ("Total Faces", "total_faces", "0"),
            ("Authorized", "authorized", "0"),
            ("Unauthorized", "unauthorized", "0"),
            ("Auth Rate", "auth_rate", "0.0%"),
            ("Detection Interval", "detection_interval", str(self.face_detection_interval))
        ])
        
        self.create_metrics_section(metrics_frame, "📈 Session Stats", [
            ("Runtime", "runtime", "0.0s"),
            ("Video Frames", "frame_count", "0"),
            ("Faces Processed", "faces_processed", "0"),
            ("Errors", "error_count", "0"),
            ("Critical Alerts", "critical_alerts", "0")
        ])
    
    def create_metrics_section(self, parent, title, metrics):
        """Create a metrics section with title and metrics."""
        section_frame = tk.Frame(parent, bg='#1e1e1e', relief='ridge', bd=1)
        section_frame.pack(fill='x', padx=10, pady=5)
        
        # Section title
        tk.Label(
            section_frame,
            text=title,
            font=('Arial', 11, 'bold'),
            fg='#ffd700',
            bg='#1e1e1e'
        ).pack(pady=5)
        
        # Metrics
        for label, var_name, default_value in metrics:
            metric_frame = tk.Frame(section_frame, bg='#1e1e1e')
            metric_frame.pack(fill='x', padx=10, pady=2)
            
            tk.Label(
                metric_frame,
                text=f"{label}:",
                font=('Arial', 9),
                fg='#ffffff',
                bg='#1e1e1e'
            ).pack(side='left')
            
            var = tk.StringVar(value=default_value)
            self.metrics_vars[var_name] = var
            
            tk.Label(
                metric_frame,
                textvariable=var,
                font=('Arial', 9, 'bold'),
                fg='#00ff00' if 'authorized' in var_name else '#ffffff',
                bg='#1e1e1e'
            ).pack(side='right')
    
    def on_camera_changed(self, event=None):
        """Handle camera selection change."""
        if self.is_running:
            messagebox.showwarning(
                "Camera Change",
                "Please stop validation before changing camera."
            )
            return
        
        selected_text = self.selected_camera.get()
        if selected_text:
            camera_index = int(selected_text.split()[-1])
            print(f"📹 Selected camera: {camera_index}")
    
    def on_fps_changed(self, event=None):
        """Handle FPS setting change."""
        try:
            new_fps = int(self.fps_var.get())
            self.target_fps = new_fps
            self.frame_interval = 1.0 / self.target_fps
            print(f"🎯 Target FPS set to: {new_fps}")
        except ValueError:
            print("❌ Invalid FPS value")
    
    def on_quality_changed(self):
        """Handle adaptive quality toggle."""
        self.adaptive_quality = self.quality_var.get()
        print(f"🎨 Adaptive quality: {'Enabled' if self.adaptive_quality else 'Disabled'}")
        if not self.adaptive_quality:
            self.current_quality_scale = 1.0
    
    def get_selected_camera_index(self):
        """Get the currently selected camera index."""
        selected_text = self.selected_camera.get()
        if selected_text:
            return int(selected_text.split()[-1])
        return self.available_cameras[0] if self.available_cameras else 0
    
    def start_validation(self):
        """Start the validation process."""
        if self.is_running:
            return
        
        try:
            camera_index = self.get_selected_camera_index()
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                messagebox.showerror("Camera Error", f"Could not open camera {camera_index}")
                return
            
            # Set camera properties for better performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            self.stop_threads = False
            self.start_time = time.time()
            
            # Update UI
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            
            # Start camera and processing threads
            self.camera_thread = threading.Thread(target=self.video_display_loop, daemon=True)
            self.face_processing_thread = threading.Thread(target=self.face_processing_loop, daemon=True)
            
            self.camera_thread.start()
            self.face_processing_thread.start()
            
            print(f"🚀 Validation started with camera {camera_index}")
            
        except Exception as e:
            messagebox.showerror("Startup Error", f"Error starting validation: {e}")
            self.stop_validation()
    
    def stop_validation(self):
        """Stop the validation process."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_threads = True
        
        # Wait for threads to finish
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2)
        if self.face_processing_thread and self.face_processing_thread.is_alive():
            self.face_processing_thread.join(timeout=2)
        
        # Release camera
        if self.camera:
            self.camera.release()
            self.camera = None
        
        # Update UI
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
        # Clear video display
        self.video_label.config(
            image='',
            text="Validation stopped\nClick 'Start Validation' to resume",
            fg='#ffffff'
        )
        
        print("🛑 Validation stopped")
    
    def reset_stats(self):
        """Reset all statistics."""
        self.fps_counter.clear()
        self.processing_times.clear()
        self.frame_count = 0
        self.total_faces = 0
        self.total_authorized = 0
        self.total_unauthorized = 0
        self.error_count = 0
        self.critical_alerts = 0
        self.faces_processed_count = 0
        self.start_time = time.time() if self.is_running else None
        
        # Reset metrics display
        self.metrics_vars['fps'].set("0.0")
        self.metrics_vars['target_fps'].set(str(self.target_fps))
        self.metrics_vars['processing_time'].set("0.000s")
        self.metrics_vars['total_faces'].set("0")
        self.metrics_vars['authorized'].set("0")
        self.metrics_vars['unauthorized'].set("0")
        self.metrics_vars['auth_rate'].set("0.0%")
        self.metrics_vars['runtime'].set("0.0s")
        self.metrics_vars['frame_count'].set("0")
        self.metrics_vars['faces_processed'].set("0")
        self.metrics_vars['error_count'].set("0")
        self.metrics_vars['critical_alerts'].set("0")
        self.metrics_vars['quality_scale'].set("100%")
        self.metrics_vars['detection_interval'].set(str(self.face_detection_interval))
        
        print("🔄 Statistics reset")
    
    def video_display_loop(self):
        """Smooth video display loop - runs at target FPS for smooth visual feed."""
        while self.is_running and not self.stop_threads:
            try:
                loop_start_time = time.time()
                
                ret, frame = self.camera.read()
                if not ret:
                    self.error_count += 1
                    continue
                
                # Store frame for face processing thread
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Always display the frame with any cached face detection results
                display_frame = self.prepare_display_frame(frame)
                
                # Convert to PIL Image and then to PhotoImage
                pil_image = Image.fromarray(display_frame)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update GUI in main thread
                self.root.after(0, self.update_video_display, photo)
                
                # Update frame count and FPS tracking
                self.frame_count += 1
                self.update_fps_tracking(loop_start_time)
                
                # Maintain smooth video FPS (no skipping for video display)
                self.maintain_video_fps(loop_start_time)
                
            except Exception as e:
                print(f"❌ Video display error: {e}")
                self.error_count += 1
                time.sleep(0.03)
    
    def face_processing_loop(self):
        """Background face processing loop - runs at optimized intervals."""
        while self.is_running and not self.stop_threads:
            try:
                # Get frame for processing
                with self.frame_lock:
                    if self.current_frame is not None:
                        processing_frame = self.current_frame.copy()
                    else:
                        time.sleep(0.1)
                        continue
                
                # Increment face detection counter
                self.face_detection_counter += 1
                
                # Only run face detection every N frames to reduce CPU load
                if self.face_detection_counter >= self.face_detection_interval:
                    self.face_detection_counter = 0
                    
                    # Check CPU usage before heavy processing
                    cpu_usage = psutil.cpu_percent(interval=0.1)
                    if cpu_usage > 80:
                        # Skip heavy processing if CPU is overloaded but continue video
                        time.sleep(0.1)
                        continue
                    
                    start_process_time = time.time()
                    
                    # Run face detection on the processing frame
                    _, face_results = self.process_frame(processing_frame)
                    self.faces_processed_count += 1  # Count actual processing operations
                    
                    # Calculate processing time
                    process_time = time.time() - start_process_time
                    self.processing_times.append(process_time)
                    
                    # Update face detection results
                    with self.processing_lock:
                        self.last_face_results = face_results
                    
                    # Update statistics
                    if face_results:
                        self.total_faces += len(face_results)
                        for result in face_results:
                            if result.get('authorized', False):
                                self.total_authorized += 1
                            else:
                                self.total_unauthorized += 1
                
                # Sleep based on face detection interval to manage CPU usage
                sleep_time = max(0.05, self.face_detection_interval * 0.033)  # Adaptive sleep
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"❌ Face processing error: {e}")
                self.error_count += 1
                time.sleep(0.1)
    
    def prepare_display_frame(self, frame):
        """Prepare frame for display with cached face detection results."""
        # Draw cached face detection results on the frame
        display_frame = frame.copy()
        
        # Get current face results
        with self.processing_lock:
            current_face_results = self.last_face_results.copy()
        
        # Draw face detection results
        for result in current_face_results:
            bbox = result.get('bbox')
            name = result.get('name', 'Unknown')
            confidence = result.get('confidence', 0.0)
            is_authorized = result.get('authorized', False)
            
            if bbox:
                x, y, w, h = bbox
                color = (0, 255, 0) if is_authorized else (0, 0, 255)
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                
                # Create label
                label = f"{name} ({confidence:.2f})"
                if is_authorized:
                    label += " ✓"
                else:
                    label += " ✗"
                
                # Draw label
                cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Convert color space and resize for display
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        display_frame = cv2.resize(display_frame, (640, 480))
        
        return display_frame
    
    def maintain_video_fps(self, loop_start_time):
        """Maintain smooth video FPS without skipping frames."""
        loop_time = time.time() - loop_start_time
        target_loop_time = self.frame_interval
        
        # Calculate required delay for target FPS
        delay_needed = target_loop_time - loop_time
        
        # Apply delay if needed to maintain target FPS
        if delay_needed > 0:
            delay_needed = min(delay_needed, 0.1)  # Max 100ms delay
            time.sleep(delay_needed)
    
    def process_frame(self, frame):
        """Process frame for face recognition."""
        if not self.face_processor:
            return frame, []
        
        try:
            # Use face processor with the correct method
            result = self.face_processor.process_image_optimized(frame, "live_camera")
            
            faces_detected = result.get('faces_detected', [])
            processed_results = []
            
            # Draw bounding boxes and labels for each detected face
            for face_info in faces_detected:
                # Extract face information
                face_data = face_info.get('face', {})
                bbox = face_data.get('bbox')
                identity = face_info.get('identity', {})
                name = identity.get('name', 'Unknown')
                confidence = identity.get('confidence', 0.0)
                is_authorized = identity.get('authorized', False)
                
                if bbox and len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    # Convert to x, y, w, h format
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    
                    # Choose color based on authorization
                    color = (0, 255, 0) if is_authorized else (0, 0, 255)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Create label
                    label = f"{name} ({confidence:.2f})"
                    if is_authorized:
                        label += " ✓"
                    else:
                        label += " ✗"
                    
                    # Draw label
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Add to processed results
                    processed_results.append({
                        'bbox': (x, y, w, h),
                        'name': name,
                        'confidence': confidence,
                        'authorized': is_authorized
                    })
            
            return frame, processed_results
            
        except Exception as e:
            print(f"❌ Face processing error: {e}")
            return frame, []
    
    def should_skip_frame(self, current_time):
        """Determine if current frame should be skipped for FPS control."""
        # Note: This method is kept for compatibility but not used in video display
        # Video display now always shows frames for smoothness
        return False
    
    def process_frame_optimized(self, frame):
        """Legacy method - now handled by separate processing thread."""
        # This method is kept for compatibility but processing is now handled
        # by the dedicated face_processing_loop for better performance
        return self.process_frame(frame)
    
    def draw_cached_results(self, frame):
        """Legacy method - now handled by prepare_display_frame."""
        # This functionality is now handled by prepare_display_frame
        return self.prepare_display_frame(frame)
    
    def optimize_frame_for_display(self, frame):
        """Legacy method - now handled by prepare_display_frame."""
        # This functionality is now integrated into prepare_display_frame
        # for better performance and smoother video
        return self.prepare_display_frame(frame)
    
    def update_fps_tracking(self, current_time):
        """Update FPS tracking with current timestamp."""
        if self.fps_counter:
            # Calculate instantaneous FPS
            time_diff = current_time - self.fps_counter[-1]
            if time_diff > 0:
                fps = 1.0 / time_diff
            else:
                fps = 0
        else:
            fps = 0
        
        self.fps_counter.append(current_time)
        self.last_frame_time = current_time
    
    def adaptive_frame_delay(self, loop_start_time, process_time):
        """Legacy method - frame delay now handled by maintain_video_fps."""
        # This functionality is now handled by maintain_video_fps for smoother video
        pass
    
    def update_video_display(self, photo):
        """Update video display in GUI."""
        try:
            self.video_label.config(image=photo, text='')
            self.video_label.image = photo  # Keep a reference
        except Exception as e:
            print(f"❌ Video display error: {e}")
    
    def start_metrics_update(self):
        """Start metrics update loop."""
        self.metrics_thread = threading.Thread(target=self.metrics_loop, daemon=True)
        self.metrics_thread.start()
    
    def metrics_loop(self):
        """Update metrics display."""
        while True:
            try:
                if self.is_running:
                    # Calculate current metrics
                    current_fps = len(self.fps_counter) / 30.0 if len(self.fps_counter) > 1 else 0
                    avg_process_time = np.mean(self.processing_times) if self.processing_times else 0
                    cpu_usage = psutil.cpu_percent()
                    memory_usage = psutil.virtual_memory().percent
                    runtime = time.time() - self.start_time if self.start_time else 0
                    auth_rate = (self.total_authorized / max(self.total_faces, 1)) * 100
                    
                    # Update metrics in GUI thread
                    self.root.after(0, self.update_metrics, {
                        'fps': f"{current_fps:.1f}",
                        'target_fps': str(self.target_fps),
                        'processing_time': f"{avg_process_time:.3f}s",
                        'cpu_usage': f"{cpu_usage:.1f}%",
                        'memory_usage': f"{memory_usage:.1f}%",
                        'quality_scale': f"{int(self.current_quality_scale * 100)}%",
                        'total_faces': str(self.total_faces),
                        'authorized': str(self.total_authorized),
                        'unauthorized': str(self.total_unauthorized),
                        'auth_rate': f"{auth_rate:.1f}%",
                        'detection_interval': str(self.face_detection_interval),
                        'runtime': f"{runtime:.1f}s",
                        'frame_count': str(self.frame_count),
                        'faces_processed': str(self.faces_processed_count),
                        'error_count': str(self.error_count),
                        'critical_alerts': str(self.critical_alerts)
                    })
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                print(f"❌ Metrics update error: {e}")
                time.sleep(1)
    
    def update_metrics(self, metrics):
        """Update metrics display in GUI."""
        try:
            for key, value in metrics.items():
                if key in self.metrics_vars:
                    self.metrics_vars[key].set(value)
        except Exception as e:
            print(f"❌ Metrics display error: {e}")
    
    def on_closing(self):
        """Handle application closing."""
        self.stop_validation()
        self.stop_threads = True
        
        # Wait a moment for threads to finish
        time.sleep(0.5)
        
        self.root.destroy()

def main():
    """Main application entry point."""
    print("🎥 GUI-Based Face Recognition Performance Validator")
    print("=" * 50)
    
    try:
        root = tk.Tk()
        app = CameraPerformanceValidator(root)
        
        # Handle window closing
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        print("✅ GUI initialized successfully")
        print("📹 Starting GUI application...")
        
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Application error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())