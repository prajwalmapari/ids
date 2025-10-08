#!/usr/bin/env python3
"""
Configuration Management Module
Centralized configuration loading and validation for the face recognition system
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration management class"""
    
    # Azure Storage Configuration
    AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    AZURE_STORAGE_ACCOUNT_NAME = os.getenv('AZURE_STORAGE_ACCOUNT_NAME', 'sakarguard')
    AZURE_CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME', 'sr001')
    AZURE_BLOB_NAME = os.getenv('AZURE_BLOB_NAME', 'authorised/authorised person/authorized_persons.json')
    AZURE_BLOB_URL = os.getenv('AZURE_BLOB_URL', 'https://sakarguard.blob.core.windows.net/sr001/authorised/authorised%20person/authorized_persons.json')
    
    # Database Configuration
    FACE_DATABASE_FILE = os.getenv('FACE_DATABASE_FILE', 'face_database.pkl')
    AUTHORIZED_CACHE_FILE = os.getenv('AUTHORIZED_CACHE_FILE', 'authorized_persons_cache.pkl')
    
    # Face Recognition Configuration
    RECOGNITION_THRESHOLD = float(os.getenv('RECOGNITION_THRESHOLD', '0.6'))
    DETECTION_SIZE_WIDTH = int(os.getenv('DETECTION_SIZE_WIDTH', '640'))
    DETECTION_SIZE_HEIGHT = int(os.getenv('DETECTION_SIZE_HEIGHT', '640'))
    
    # InsightFace Model Configuration
    INSIGHTFACE_MODEL_NAME = os.getenv('INSIGHTFACE_MODEL_NAME', 'buffalo_l')
    INSIGHTFACE_PROVIDERS = os.getenv('INSIGHTFACE_PROVIDERS', 'CPUExecutionProvider')
    
    # Security Configuration
    DEFAULT_ACCESS_LEVEL = os.getenv('DEFAULT_ACCESS_LEVEL', 'MEDIUM')
    DEFAULT_DEPARTMENT = os.getenv('DEFAULT_DEPARTMENT', 'Security')
    DEFAULT_STATUS = os.getenv('DEFAULT_STATUS', 'active')
    
    # Application Configuration
    MAX_FACES_PER_IMAGE = int(os.getenv('MAX_FACES_PER_IMAGE', '50'))
    VIDEO_FRAME_SKIP = int(os.getenv('VIDEO_FRAME_SKIP', '3'))
    OUTPUT_IMAGE_QUALITY = int(os.getenv('OUTPUT_IMAGE_QUALITY', '95'))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'face_recognition.log')
    
    # API Configuration
    API_HOST = os.getenv('API_HOST', 'localhost')
    API_PORT = int(os.getenv('API_PORT', '8000'))
    API_SECRET_KEY = os.getenv('API_SECRET_KEY', 'your-secret-api-key-here')
    
    # Email Configuration
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    SMTP_USERNAME = os.getenv('SMTP_USERNAME')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
    ALERT_EMAIL = os.getenv('ALERT_EMAIL')
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check critical configuration
        if not cls.AZURE_STORAGE_CONNECTION_STRING:
            validation_result['errors'].append('AZURE_STORAGE_CONNECTION_STRING is not set')
            validation_result['valid'] = False
        
        # Check optional but recommended configuration
        if not cls.SMTP_USERNAME or not cls.SMTP_PASSWORD:
            validation_result['warnings'].append('Email configuration incomplete - notifications disabled')
        
        if cls.API_SECRET_KEY == 'your-secret-api-key-here':
            validation_result['warnings'].append('Default API secret key detected - please change for production')
        
        # Validate numeric ranges
        if not (0.0 <= cls.RECOGNITION_THRESHOLD <= 1.0):
            validation_result['errors'].append('RECOGNITION_THRESHOLD must be between 0.0 and 1.0')
            validation_result['valid'] = False
        
        if cls.DETECTION_SIZE_WIDTH < 100 or cls.DETECTION_SIZE_HEIGHT < 100:
            validation_result['warnings'].append('Detection size seems too small, may affect accuracy')
        
        return validation_result
    
    @classmethod
    def get_azure_config(cls) -> Dict[str, str]:
        """Get Azure-specific configuration"""
        return {
            'connection_string': cls.AZURE_STORAGE_CONNECTION_STRING,
            'account_name': cls.AZURE_STORAGE_ACCOUNT_NAME,
            'container_name': cls.AZURE_CONTAINER_NAME,
            'blob_name': cls.AZURE_BLOB_NAME,
            'blob_url': cls.AZURE_BLOB_URL
        }
    
    @classmethod
    def get_face_recognition_config(cls) -> Dict[str, Any]:
        """Get face recognition specific configuration"""
        return {
            'model_name': cls.INSIGHTFACE_MODEL_NAME,
            'providers': [cls.INSIGHTFACE_PROVIDERS],
            'detection_size': (cls.DETECTION_SIZE_WIDTH, cls.DETECTION_SIZE_HEIGHT),
            'threshold': cls.RECOGNITION_THRESHOLD,
            'max_faces': cls.MAX_FACES_PER_IMAGE
        }
    
    @classmethod
    def get_security_config(cls) -> Dict[str, str]:
        """Get security-specific configuration"""
        return {
            'default_access_level': cls.DEFAULT_ACCESS_LEVEL,
            'default_department': cls.DEFAULT_DEPARTMENT,
            'default_status': cls.DEFAULT_STATUS
        }
    
    @classmethod
    def print_config_summary(cls):
        """Print configuration summary"""
        print("=" * 50)
        print("FACE RECOGNITION SYSTEM CONFIGURATION")
        print("=" * 50)
        
        print(f"Azure Storage Account: {cls.AZURE_STORAGE_ACCOUNT_NAME}")
        print(f"Container: {cls.AZURE_CONTAINER_NAME}")
        print(f"Recognition Threshold: {cls.RECOGNITION_THRESHOLD}")
        print(f"Detection Size: {cls.DETECTION_SIZE_WIDTH}x{cls.DETECTION_SIZE_HEIGHT}")
        print(f"Model: {cls.INSIGHTFACE_MODEL_NAME}")
        print(f"Default Access Level: {cls.DEFAULT_ACCESS_LEVEL}")
        
        # Validation
        validation = cls.validate_config()
        if validation['valid']:
            print("✅ Configuration: VALID")
        else:
            print("❌ Configuration: INVALID")
            for error in validation['errors']:
                print(f"   ERROR: {error}")
        
        if validation['warnings']:
            print("⚠️  Warnings:")
            for warning in validation['warnings']:
                print(f"   WARNING: {warning}")
        
        print("=" * 50)

def load_config() -> Config:
    """Load and return configuration"""
    return Config()

if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    config.print_config_summary()
    
    # Test validation
    validation = config.validate_config()
    print(f"\nValidation Result: {validation}")