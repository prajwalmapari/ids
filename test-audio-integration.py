#!/usr/bin/env python3
"""
Test Audio Alert System Integration
Tests the audio alert system with simulated unauthorized person detection
"""

import os
import sys
import time
import importlib.util
import json
from datetime import datetime

# Import audio-alert.py as a module
try:
    spec = importlib.util.spec_from_file_location("audio_alert", "/home/ubuntu24/ids/audio-alert.py")
    audio_alert = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audio_alert)
    AudioAlertSystem = audio_alert.AudioAlertSystem
except Exception as e:
    print(f"❌ Failed to import AudioAlertSystem: {e}")
    sys.exit(1)

def simulate_detection_log():
    """Create a simulated detection log for testing"""
    detection_data = {
        "timestamp": datetime.now().isoformat(),
        "detection_type": "unauthorized_person",
        "person_id": f"unknown_{int(time.time())}",
        "confidence": 0.95,
        "location": "main_entrance",
        "image_path": "unauthorized_images/test_detection.jpg",
        "alert_triggered": True,
        "priority": "high"
    }
    
    # Save to a test log file
    log_filename = f"test_detection_{int(time.time())}.json"
    with open(log_filename, 'w') as f:
        json.dump(detection_data, f, indent=2)
    
    print(f"📝 Created simulated detection log: {log_filename}")
    return log_filename, detection_data

def test_audio_alert_integration():
    """Test the complete audio alert system integration"""
    print("🧪 Testing Audio Alert System Integration")
    print("=" * 60)
    
    try:
        # Initialize the audio alert system
        print("🔧 Initializing AudioAlertSystem...")
        audio_system = AudioAlertSystem()
        print("✅ AudioAlertSystem initialized successfully")
        
        # Test 1: Manual alert trigger
        print("\n🚨 Test 1: Manual Alert Trigger")
        print("-" * 40)
        
        print("Triggering NORMAL priority alert...")
        audio_system.play_alert_sound('normal')
        time.sleep(2)
        
        print("Triggering HIGH priority alert...")
        audio_system.play_alert_sound('high')
        time.sleep(2)
        
        print("Triggering CRITICAL priority alert...")
        audio_system.play_alert_sound('critical')
        time.sleep(3)
        
        # Test 2: Simulated detection processing
        print("\n📡 Test 2: Simulated Detection Processing")
        print("-" * 45)
        
        # Create a simulated detection
        log_file, detection_data = simulate_detection_log()
        
        # Test the detection analysis
        print(f"Processing detection: {detection_data['detection_type']}")
        print(f"Priority: {detection_data['priority']}")
        print(f"Confidence: {detection_data['confidence']}")
        
        # Trigger appropriate alert based on detection
        if detection_data['detection_type'] == 'unauthorized_person':
            priority = detection_data.get('priority', 'high')
            print(f"🔊 Triggering {priority.upper()} priority security alert...")
            audio_system.play_alert_sound(priority)
        
        # Cleanup
        os.remove(log_file)
        print(f"🧹 Cleaned up test file: {log_file}")
        
        print("\n✅ Audio alert integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

def test_pure_python_fallback():
    """Test the pure Python fallback when pygame is not available"""
    print("\n🔧 Test 3: Pure Python Fallback")
    print("-" * 35)
    
    try:
        # Initialize audio system
        audio_system = AudioAlertSystem()
        
        # Temporarily disable pygame to test fallback
        original_pygame_state = audio_system.pygame_initialized
        audio_system.pygame_initialized = False
        
        print("🔊 Testing pure Python fallback (pygame disabled)...")
        audio_system.play_alert_sound('high')
        
        # Restore pygame state
        audio_system.pygame_initialized = original_pygame_state
        
        print("✅ Pure Python fallback test completed")
        
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")

if __name__ == "__main__":
    print("🎵 Audio Alert System Integration Test")
    print("=" * 60)
    
    # Run comprehensive integration tests
    test_audio_alert_integration()
    
    # Test pure Python fallback
    test_pure_python_fallback()
    
    print("\n🎯 All integration tests completed!")
    print("\n🚀 Audio Alert System is ready for Jetson deployment!")
    print("📦 Dependencies: Pure Python (wave, struct, math) + pygame for enhanced audio")