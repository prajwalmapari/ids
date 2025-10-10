#!/usr/bin/env python3
"""
Azure-Based Audio Alert System for Unauthorized Person Detection
Monitors Azure detection logs and generates audio alerts for security events
"""

import os
import sys
import json
import time
import math
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import logging
from io import StringIO

# Audio libraries
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️ pyttsx3 not available. Install with: pip install pyttsx3")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️ pygame not available. Install with: pip install pygame")

# Azure SDK
try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import ResourceNotFoundError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("❌ Azure SDK not available. Install with: pip install azure-storage-blob")
    sys.exit(1)

# Environment variables
from dotenv import load_dotenv
load_dotenv()

class AudioAlertSystem:
    """
    Azure-based audio alert system for unauthorized person detection
    """
    
    def __init__(self):
        """Initialize the audio alert system"""
        self.setup_logging()
        self.setup_azure()
        self.setup_audio()
        self.setup_monitoring()
        
        # Alert configuration
        self.alert_config = {
            'unauthorized_person': {
                'enabled': True,
                'cooldown': 10,  # seconds between same alerts
                'priority': 'high',
                'message': 'Unauthorized person detected in restricted area'
            },
            'multiple_unauthorized': {
                'enabled': True,
                'cooldown': 15,
                'priority': 'critical',
                'message': 'Multiple unauthorized persons detected. Security alert!'
            },
            'continuous_detection': {
                'enabled': True,
                'cooldown': 30,
                'priority': 'high',
                'message': 'Continuous unauthorized activity detected'
            }
        }
        
        # Tracking variables
        self.last_alerts: Dict[str, datetime] = {}
        self.detection_history: List[Dict] = []
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.WARNING,  # Only show warnings and errors
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('audio_alerts.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Disable Azure SDK verbose logging
        logging.getLogger('azure').setLevel(logging.ERROR)
        logging.getLogger('azure.storage').setLevel(logging.ERROR)
        logging.getLogger('azure.core').setLevel(logging.ERROR)
        
    def setup_azure(self):
        """Setup Azure blob storage client"""
        try:
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            if not connection_string:
                raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found in environment")
                
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            self.container_name = os.getenv('AZURE_CONTAINER_NAME', 'sr001')
            self.log_blob_path = os.getenv('AZURE_LOG_BLOB', 'unauthorised_person/detection_logs/')
            
            print(f"✅ Azure client initialized - Container: {self.container_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup Azure client: {e}")
            sys.exit(1)
            
    def setup_audio(self):
        """Setup audio systems (TTS and sound effects)"""
        self.tts_engine = None
        self.pygame_initialized = False
        
        # Setup Text-to-Speech
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                # Configure TTS settings
                self.tts_engine.setProperty('rate', 150)  # Speed
                self.tts_engine.setProperty('volume', 0.9)  # Volume
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    # Use first available voice
                    self.tts_engine.setProperty('voice', voices[0].id)
                print("✅ Text-to-Speech engine initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ TTS initialization failed: {e}")
                self.tts_engine = None
        
        # Setup Pygame for sound effects
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                self.pygame_initialized = True
                print("✅ Pygame audio system initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Pygame initialization failed: {e}")
                
        # Create beep sound files for reliable audio
        self.create_beep_files()
        
    def create_beep_files(self):
        """Create WAV beep files using pure Python"""
        try:
            import wave
            import struct
            import math
            
            # Create beep files if they don't exist
            beep_configs = {
                'beep_critical.wav': (1000, 0.3, 44100),  # High pitch, short
                'beep_high.wav': (800, 0.4, 44100),       # Medium pitch, medium
                'beep_normal.wav': (600, 0.5, 44100)      # Low pitch, long
            }
            
            for filename, (freq, duration, sample_rate) in beep_configs.items():
                if not os.path.exists(filename):
                    self._create_beep_wav(filename, freq, duration, sample_rate)
                    
            print("✅ Audio beep files created with pure Python")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create beep files: {e}")
            
    def _create_beep_wav(self, filename: str, frequency: int, duration: float, sample_rate: int = 44100):
        """Create a WAV beep file using pure Python"""
        import wave
        import struct
        import math
        
        # Calculate number of frames
        frames = int(duration * sample_rate)
        
        # Generate sine wave data
        audio_data = []
        for i in range(frames):
            # Generate sine wave sample
            t = float(i) / sample_rate
            sample = int(32767 * 0.9 * math.sin(2 * math.pi * frequency * t))
            # Create stereo (duplicate for left and right channels)
            audio_data.append(struct.pack('<hh', sample, sample))
        
        # Write WAV file
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(2)  # Stereo
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b''.join(audio_data))
                
    def setup_monitoring(self):
        """Setup detection monitoring parameters"""
        self.current_date = datetime.now().strftime('%Y-%m-%d')
        self.log_blob_name = f"{self.log_blob_path}detection_log_{self.current_date}.json"
        self.last_processed_entry = 0
        self.last_log_check = datetime.now()
        
    def get_latest_detection_log(self) -> Optional[Dict]:
        """Fetch the latest detection log from Azure"""
        try:
            # Update current date and blob name
            current_date = datetime.now().strftime('%Y-%m-%d')
            if current_date != self.current_date:
                self.current_date = current_date
                self.log_blob_name = f"{self.log_blob_path}detection_log_{self.current_date}.json"
                self.last_processed_entry = 0
                
            # Try current date first
            for days_back in range(3):  # Try current day and 2 days back
                try_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                try_blob_name = f"{self.log_blob_path}detection_log_{try_date}.json"
                
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name, 
                    blob=try_blob_name
                )
                
                # Download and parse the log
                blob_data = blob_client.download_blob().readall()
                
                # Check if blob has content
                if len(blob_data) == 0:
                    continue  # Skip empty logs silently
                    
                try:
                    log_data = json.loads(blob_data.decode('utf-8'))
                    
                    # Validate log structure
                    if not log_data or 'detections' not in log_data:
                        continue  # Skip invalid logs silently
                        
                    # If this is not today's log, reset processed entry count
                    if try_date != self.current_date:
                        self.current_date = try_date
                        self.log_blob_name = try_blob_name
                        self.last_processed_entry = 0
                        print(f"📅 Using detection log from: {try_date}")
                        
                    return log_data
                    
                except json.JSONDecodeError as e:
                    continue  # Skip logs with JSON errors silently
                    
        except ResourceNotFoundError:
            pass  # Log not found is normal, don't log it
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch detection log: {e}")
            return None
            
        # Don't log when no detection logs found - this is normal
        return None
            
    def analyze_detections(self, log_data: Dict) -> List[Dict]:
        """Analyze detection log for alert-worthy events"""
        if not log_data or 'detections' not in log_data:
            return []
            
        detections = log_data['detections']
        new_detections = detections[self.last_processed_entry:]
        
        if not new_detections:
            return []
            
        self.last_processed_entry = len(detections)
        alerts_to_trigger = []
        
        # Analyze recent detections (last 30 seconds)
        current_time = datetime.now()
        recent_detections = []
        
        for detection in new_detections:
            try:
                detection_time = datetime.fromisoformat(detection['timestamp'].replace('Z', '+00:00'))
                if (current_time - detection_time.replace(tzinfo=None)).total_seconds() <= 30:
                    recent_detections.append(detection)
            except:
                recent_detections.append(detection)  # Include if timestamp parsing fails
                
        # Count unauthorized detections
        unauthorized_count = sum(1 for d in recent_detections if d.get('status') == 'UNAUTHORIZED')
        
        if unauthorized_count > 0:
            # Single unauthorized person alert
            if unauthorized_count == 1:
                alerts_to_trigger.append({
                    'type': 'unauthorized_person',
                    'count': 1,
                    'detections': recent_detections
                })
            
            # Multiple unauthorized persons alert
            elif unauthorized_count > 1:
                alerts_to_trigger.append({
                    'type': 'multiple_unauthorized',
                    'count': unauthorized_count,
                    'detections': recent_detections
                })
                
        # Check for continuous detection pattern
        if len(self.detection_history) >= 5:
            # Check if we have consistent unauthorized detections
            recent_unauthorized = sum(1 for d in self.detection_history[-5:] if d.get('has_unauthorized', False))
            if recent_unauthorized >= 4:
                alerts_to_trigger.append({
                    'type': 'continuous_detection',
                    'count': recent_unauthorized,
                    'detections': recent_detections
                })
                
        # Update detection history
        self.detection_history.append({
            'timestamp': current_time,
            'unauthorized_count': unauthorized_count,
            'has_unauthorized': unauthorized_count > 0
        })
        
        # Keep only last 10 entries
        if len(self.detection_history) > 10:
            self.detection_history = self.detection_history[-10:]
            
        return alerts_to_trigger
        
    def should_trigger_alert(self, alert_type: str) -> bool:
        """Check if enough time has passed since last alert of this type"""
        if alert_type not in self.alert_config:
            return False
            
        if not self.alert_config[alert_type]['enabled']:
            return False
            
        last_alert = self.last_alerts.get(alert_type)
        if not last_alert:
            return True
            
        cooldown = self.alert_config[alert_type]['cooldown']
        time_diff = (datetime.now() - last_alert).total_seconds()
        
        return time_diff >= cooldown
        
    def play_text_to_speech(self, message: str):
        """Play text-to-speech alert"""
        if not self.tts_engine:
            return False
            
        try:
            self.tts_engine.say(message)
            self.tts_engine.runAndWait()
            return True
        except Exception as e:
            self.logger.error(f"❌ TTS playback failed: {e}")
            return False
            
    def play_alert_sound(self, priority: str = 'high'):
        """Play alert sound effect using pure Python"""
        try:
            print(f"🔊 Playing {priority.upper()} priority beep sound...")
            
            # Use pygame to play pre-generated WAV files
            if not self.pygame_initialized:
                print("   🔄 Pygame not available, using pure Python beep generation...")
                return self._play_pure_python_beep(priority)
            
            if priority == 'critical':
                # Critical: 5 rapid beeps
                print("   🚨 CRITICAL: Playing 5 rapid beeps at 1000Hz...")
                for i in range(5):
                    print(f"   🔊 Beep {i+1}/5")
                    self._play_wav_with_pygame('beep_critical.wav')
                    if i < 4:  # Don't sleep after last beep
                        time.sleep(0.1)
            elif priority == 'high':
                # High: 3 beeps
                print("   🔊 HIGH: Playing 3 beeps at 800Hz...")
                for i in range(3):
                    print(f"   🔊 Beep {i+1}/3")
                    self._play_wav_with_pygame('beep_high.wav')
                    if i < 2:  # Don't sleep after last beep
                        time.sleep(0.2)
            else:
                # Normal: 2 beeps
                print("   🔊 NORMAL: Playing 2 beeps at 600Hz...")
                for i in range(2):
                    print(f"   🔊 Beep {i+1}/2")
                    self._play_wav_with_pygame('beep_normal.wav')
                    if i < 1:  # Don't sleep after last beep
                        time.sleep(0.3)
            
            print("   ✅ Beep sequence completed")
            return True
        except Exception as e:
            self.logger.error(f"❌ Sound playback failed: {e}")
            # Fallback to pure Python beep
            return self._play_pure_python_beep(priority)
            
    def _play_wav_with_pygame(self, filename: str):
        """Play WAV file using pygame"""
        try:
            if not os.path.exists(filename):
                print(f"   ⚠️ WAV file {filename} not found")
                return False
                
            sound = pygame.mixer.Sound(filename)
            sound.set_volume(1.0)  # Maximum volume
            channel = sound.play()
            
            # Wait for sound to finish
            while channel.get_busy():
                pygame.time.wait(10)
                
            return True
        except Exception as e:
            print(f"   ⚠️ Pygame playback failed: {e}")
            return False
            
    def _play_pure_python_beep(self, priority: str = 'high'):
        """Pure Python beep generation fallback"""
        try:
            import math
            
            print(f"   🔊 Pure Python beep generation for {priority.upper()} priority...")
            
            # Configure beep parameters based on priority
            if priority == 'critical':
                beep_count, freq, duration = 5, 1000, 0.3
            elif priority == 'high':
                beep_count, freq, duration = 3, 800, 0.4
            else:
                beep_count, freq, duration = 2, 600, 0.5
            
            for i in range(beep_count):
                print(f"   🔊 Pure Python Beep {i+1}/{beep_count}")
                
                # Generate beep using numpy if available
                try:
                    import numpy as np
                    sample_rate = 44100
                    frames = int(duration * sample_rate)
                    t = np.linspace(0, duration, frames, False)
                    wave = np.sin(2 * np.pi * freq * t)
                    wave = (wave * 32767 * 0.9).astype(np.int16)
                    stereo_wave = np.column_stack((wave, wave))
                    stereo_wave = np.ascontiguousarray(stereo_wave)
                    
                    if self.pygame_initialized:
                        sound = pygame.sndarray.make_sound(stereo_wave)
                        sound.set_volume(1.0)
                        sound.play()
                        pygame.time.wait(int(duration * 1000))
                    else:
                        # Last resort: terminal beep
                        print('\a', end='', flush=True)
                        
                except ImportError:
                    # Final fallback: terminal beep
                    print('\a', end='', flush=True)
                    time.sleep(duration)
                
                if i < beep_count - 1:
                    time.sleep(0.1)
            
            print("   ✅ Pure Python beep sequence completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Pure Python beep failed: {e}")
            # Ultimate fallback
            for _ in range(3):
                print('\a', end='', flush=True)
                time.sleep(0.2)
            return False
            
    def generate_beep(self, frequency: int, duration: float):
        """Generate a beep sound at specified frequency and duration"""
        if not self.pygame_initialized:
            return
            
        try:
            import numpy as np
            
            sample_rate = 22050
            frames = int(duration * sample_rate)
            
            # Generate sine wave
            t = np.linspace(0, duration, frames, False)
            wave = np.sin(2 * np.pi * frequency * t)
            
            # Convert to 16-bit integers with maximum volume
            wave = (wave * 32767 * 0.9).astype(np.int16)  # 90% max volume for loud beeps
            
            # Create stereo array and ensure it's C-contiguous
            stereo_wave = np.column_stack((wave, wave))
            stereo_wave = np.ascontiguousarray(stereo_wave)
            
            sound = pygame.sndarray.make_sound(stereo_wave)
            sound.set_volume(1.0)  # Maximum volume
            sound.play()
            pygame.time.wait(int(duration * 1000))  # Use pygame wait instead of time.sleep
            
        except ImportError:
            # Fallback to simple beep without numpy
            self._generate_simple_beep(frequency, duration)
                
        except Exception as e:
            self.logger.warning(f"⚠️ NumPy beep generation failed: {e}, trying fallback")
            self._generate_simple_beep(frequency, duration)
            
    def _generate_simple_beep(self, frequency: int, duration: float):
        """Simple beep generation without numpy"""
        try:
            sample_rate = 22050
            frames = int(duration * sample_rate)
            
            # Generate simple sine wave without numpy - louder
            wave_data = []
            for i in range(frames):
                t = float(i) / sample_rate
                sample = int(32767 * 0.9 * math.sin(2 * math.pi * frequency * t))  # 90% max volume
                wave_data.append((sample, sample))
                
            sound = pygame.sndarray.make_sound(wave_data)
            sound.set_volume(1.0)  # Maximum volume
            sound.play()
            pygame.time.wait(int(duration * 1000))  # Use pygame wait
            
        except Exception as e:
            self.logger.warning(f"⚠️ Simple beep generation also failed: {e}")
            # Just play multiple system beeps as last resort for loud sound
            for _ in range(3):
                print('\a', end='', flush=True)
                time.sleep(0.1)
            
    def trigger_alert(self, alert_data: Dict):
        """Trigger an audio alert based on detection data"""
        alert_type = alert_data['type']
        
        if not self.should_trigger_alert(alert_type):
            return
            
        config = self.alert_config[alert_type]
        priority = config['priority']
        count = alert_data.get('count', 1)
        is_security_alert = alert_data.get('is_security_alert', True)  # Real security alerts vs test alerts
        
        print(f"🚨 SECURITY ALERT: {alert_type} - {count} unauthorized person(s) detected")
        
        # Only play beep sounds for real security alerts (no TTS voice)
        if is_security_alert:
            threading.Thread(target=self.play_alert_sound, args=(priority,), daemon=True).start()
        
        # Record alert timestamp
        self.last_alerts[alert_type] = datetime.now()
        
        # Log alert details
        print(f"📊 Alert Details: Type={alert_type}, Count={count}, Priority={priority}")
        
    def monitor_loop(self):
        """Main monitoring loop"""
        print("🔍 Starting audio alert monitoring...")
        
        while self.running:
            try:
                # Fetch latest detection log
                log_data = self.get_latest_detection_log()
                
                if log_data:
                    # Analyze for alert-worthy events
                    alerts = self.analyze_detections(log_data)
                    
                    # Trigger alerts (these are real security alerts)
                    for alert in alerts:
                        alert['is_security_alert'] = True  # Mark as real security alert
                        self.trigger_alert(alert)
                        
                # Wait before next check
                time.sleep(2)  # Check every 2 seconds
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"❌ Monitor loop error: {e}")
                time.sleep(5)  # Wait longer on error
                
        print("🛑 Audio alert monitoring stopped")
        
    def start_monitoring(self):
        """Start the monitoring system"""
        if self.running:
            print("⚠️ Monitoring already running")
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("🚀 Audio alert system started")
        
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        print("🛑 Audio alert system stopped")
        
    def show_status(self):
        """Display current system status"""
        print("\n" + "="*60)
        print("🔊 AUDIO ALERT SYSTEM STATUS")
        print("="*60)
        print(f"📊 System Status: {'🟢 RUNNING' if self.running else '🔴 STOPPED'}")
        print(f"🔗 Azure Container: {self.container_name}")
        print(f"📝 Current Log: {self.log_blob_name}")
        print(f"🔢 Processed Entries: {self.last_processed_entry}")
        print(f"🎵 TTS Available: {'✅' if self.tts_engine else '❌'}")
        print(f"🔊 Sound Effects: {'✅' if self.pygame_initialized else '❌'}")
        
        print("\n📋 Alert Configuration:")
        for alert_type, config in self.alert_config.items():
            status = "🟢 ENABLED" if config['enabled'] else "🔴 DISABLED"
            print(f"   {alert_type}: {status} (Cooldown: {config['cooldown']}s)")
            
        print("\n🕐 Last Alerts:")
        if self.last_alerts:
            for alert_type, timestamp in self.last_alerts.items():
                print(f"   {alert_type}: {timestamp.strftime('%H:%M:%S')}")
        else:
            print("   No alerts triggered yet")
            
        print("\n📈 Detection History (Last 10):")
        for i, entry in enumerate(self.detection_history[-5:], 1):
            timestamp = entry['timestamp'].strftime('%H:%M:%S')
            count = entry['unauthorized_count']
            print(f"   {i}. {timestamp}: {count} unauthorized detection(s)")
            
        print("="*60 + "\n")

def main():
    """Main function"""
    print("🔊 Azure-Based Audio Alert System")
    print("=" * 50)
    
    # Initialize system
    alert_system = AudioAlertSystem()
    
    try:
        # Start monitoring
        alert_system.start_monitoring()
        
        print("\n🎯 Commands:")
        print("  's' - Show status")
        print("  'q' - Quit")
        print("  't' - Test alert with beep")
        print("  'v' - Volume test (play beep file)")
        print("  'f' - Force check recent detections")
        print("\n🔍 Monitoring Azure detection logs for unauthorized persons...")
        
        while True:
            command = input("\n> ").strip().lower()
            
            if command == 'q':
                break
            elif command == 's':
                alert_system.show_status()
            elif command == 't':
                # Test alert (now with beep sound for testing)
                print("🔊 Testing audio alert with beep sound...")
                test_alert = {
                    'type': 'unauthorized_person',
                    'count': 1,
                    'detections': [{'status': 'UNAUTHORIZED', 'timestamp': datetime.now().isoformat()}],
                    'is_security_alert': True  # Enable beep for testing
                }
                alert_system.trigger_alert(test_alert)
            elif command == 'v':
                # Volume test - play beep file directly
                print("🔊 Volume test - playing beep sound...")
                os.system('aplay beep_high.wav')
                print("✅ Volume test completed")
            elif command == 'f':
                # Force check recent detections
                print("🔍 Force checking recent detections...")
                log_data = alert_system.get_latest_detection_log()
                if log_data:
                    # Temporarily reset last processed to trigger alerts
                    original_processed = alert_system.last_processed_entry
                    alert_system.last_processed_entry = max(0, len(log_data.get('detections', [])) - 5)  # Check last 5
                    alerts = alert_system.analyze_detections(log_data)
                    if alerts:
                        for alert in alerts:
                            alert['is_security_alert'] = True
                            alert_system.trigger_alert(alert)
                        print(f"✅ Triggered {len(alerts)} alert(s)")
                    else:
                        print("ℹ️ No alerts triggered from recent detections")
                        # Reset to original
                        alert_system.last_processed_entry = original_processed
                else:
                    print("❌ No detection logs found")
            else:
                print("Unknown command. Use 's' for status, 't' for test, 'v' for volume test, 'f' for force check, 'q' to quit.")
                
    except KeyboardInterrupt:
        pass
    finally:
        print("\n🛑 Shutting down audio alert system...")
        alert_system.stop_monitoring()
        print("✅ Audio alert system stopped")

if __name__ == "__main__":
    main()
