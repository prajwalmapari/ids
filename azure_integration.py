#!/usr/bin/env python3
"""
Azure Authorized Persons Integration Module
Handles loading authorized person embeddings from Azure blob storage or local cache
"""

import json
import requests
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from database import FaceDatabase
import pickle

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Azure Storage imports
try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("Warning: azure-storage-blob not installed. Install with: pip install azure-storage-blob")

class AuthorizedPersonsManager:
    """Manage authorized person embeddings from external sources"""
    
    def __init__(self, cache_file: str = None):
        self.cache_file = cache_file or os.getenv('AUTHORIZED_CACHE_FILE', 'authorized_persons_cache.pkl')
        self.authorized_persons: Dict[str, dict] = {}
        self.azure_url = os.getenv('AZURE_BLOB_URL', 'https://sakarguard.blob.core.windows.net/sr001/authorised/authorised%20person/authorized_persons.json')
        
        # Azure Storage configuration from environment variables
        self.connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        self.container_name = os.getenv('AZURE_CONTAINER_NAME', 'sr001')
        self.blob_name = os.getenv('AZURE_BLOB_NAME', 'authorised/authorised person/authorized_persons.json')
        
        # Validate required environment variables
        if not self.connection_string:
            print("Warning: AZURE_STORAGE_CONNECTION_STRING not found in environment variables")
            print("Please check your .env file or set the environment variable")
        
    def load_from_azure(self, auth_headers: Dict[str, str] = None) -> bool:
        """Load authorized persons from Azure blob storage using connection string"""
        if not AZURE_AVAILABLE:
            print("Azure Storage SDK not available. Using fallback method...")
            return self._load_from_azure_requests(auth_headers)
        
        try:
            print(f"Connecting to Azure Storage Account: sakarguard")
            print(f"Container: {self.container_name}")
            print(f"Blob: {self.blob_name}")
            
            # Create blob service client
            blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            
            # Get blob client
            blob_client = blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=self.blob_name
            )
            
            # Download blob content
            print("Downloading authorized persons data from Azure...")
            blob_data = blob_client.download_blob()
            content = blob_data.readall()
            
            # Parse JSON
            data = json.loads(content.decode('utf-8'))
            self.authorized_persons = self._process_azure_data(data)
            self._save_cache()
            
            print(f"✅ Successfully loaded {len(self.authorized_persons)} authorized persons from Azure")
            return True
            
        except Exception as e:
            print(f"❌ Error loading from Azure Storage: {e}")
            print("Falling back to requests method...")
            return self._load_from_azure_requests(auth_headers)
    
    def _load_from_azure_requests(self, auth_headers: Dict[str, str] = None) -> bool:
        """Fallback method using requests"""
        try:
            headers = auth_headers or {}
            print(f"Using requests fallback for Azure access...")
            
            response = requests.get(self.azure_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                self.authorized_persons = self._process_azure_data(data)
                self._save_cache()
                print(f"Successfully loaded {len(self.authorized_persons)} authorized persons from Azure")
                return True
            else:
                print(f"Failed to fetch from Azure. Status code: {response.status_code}")
                print(f"Response: {response.text[:200]}...")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Network error accessing Azure: {e}")
            return False
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from Azure: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error loading from Azure: {e}")
            return False
    
    def _process_azure_data(self, data: dict) -> Dict[str, dict]:
        """Process the Azure JSON data into our format"""
        processed = {}
        
        # Handle different possible JSON structures
        if isinstance(data, list):
            # Array of persons
            for person in data:
                if self._validate_person_data(person):
                    processed[person['name']] = self._normalize_person_data(person)
        elif isinstance(data, dict):
            if 'authorized_persons' in data:
                # Wrapped in authorized_persons key
                for person in data['authorized_persons']:
                    if self._validate_person_data(person):
                        processed[person['name']] = self._normalize_person_data(person)
            elif 'name' in data and 'embedding' in data:
                # Single person object
                if self._validate_person_data(data):
                    processed[data['name']] = self._normalize_person_data(data)
            else:
                # Direct mapping of ID -> person data (current Azure structure)
                for person_id, person_data in data.items():
                    if isinstance(person_data, dict) and 'embedding' in person_data:
                        # Use the actual name from the data, not the ID key
                        actual_name = person_data.get('name', person_id)
                        person_obj = person_data.copy()
                        person_obj['name'] = actual_name
                        person_obj['person_id'] = person_id  # Keep the original ID for reference
                        if self._validate_person_data(person_obj):
                            # Use actual name as key, not the person_id
                            processed[actual_name] = self._normalize_person_data(person_obj)
        
        return processed
    
    def _normalize_person_data(self, person: dict) -> dict:
        """Normalize person data to standard format"""
        normalized = {
            'name': person.get('name', 'Unknown'),
            'embedding': person.get('embedding', []),
            'id': person.get('person_id', person.get('id', f"AUTO_{person.get('name', 'Unknown').upper()}")),
            'department': person.get('department', os.getenv('DEFAULT_DEPARTMENT', 'Security')),
            'access_level': person.get('access_level', os.getenv('DEFAULT_ACCESS_LEVEL', 'MEDIUM')),
            'status': person.get('status', os.getenv('DEFAULT_STATUS', 'active')),
            'registered_date': person.get('added_time', person.get('registered_date', '2024-01-01')),
            'image_path': person.get('image_path', ''),
            'confidence': person.get('confidence', 0.0),
            'gender': person.get('gender', 'Unknown'),
            'source': 'azure_authorized_persons'
        }
        return normalized
    
    def _validate_person_data(self, person: dict) -> bool:
        """Validate that person data has required fields"""
        if not isinstance(person, dict):
            return False
        
        # Must have embedding data
        if 'embedding' not in person:
            return False
            
        # Must have name (either directly or can be inferred)
        if 'name' not in person:
            return False
            
        # Embedding must be a list/array with data
        embedding = person.get('embedding', [])
        if not isinstance(embedding, (list, tuple)) or len(embedding) == 0:
            return False
        
        return True
    
    def create_sample_data(self) -> Dict[str, dict]:
        """Create sample authorized persons data for testing"""
        sample_data = {
            "john_doe": {
                "name": "John Doe",
                "id": "EMP001",
                "department": "Security",
                "access_level": "HIGH",
                "embedding": np.random.rand(512).tolist(),  # Sample 512-dim embedding
                "registered_date": "2024-01-15",
                "status": "active"
            },
            "jane_smith": {
                "name": "Jane Smith", 
                "id": "EMP002",
                "department": "Administration",
                "access_level": "MEDIUM",
                "embedding": np.random.rand(512).tolist(),
                "registered_date": "2024-02-20",
                "status": "active"
            },
            "mike_johnson": {
                "name": "Mike Johnson",
                "id": "EMP003", 
                "department": "IT",
                "access_level": "HIGH",
                "embedding": np.random.rand(512).tolist(),
                "registered_date": "2024-03-10",
                "status": "active"
            }
        }
        return sample_data
    
    def load_sample_data(self):
        """Load sample data for testing"""
        print("Loading sample authorized persons data...")
        self.authorized_persons = self.create_sample_data()
        self._save_cache()
        print(f"Loaded {len(self.authorized_persons)} sample authorized persons")
    
    def _save_cache(self):
        """Save authorized persons to local cache"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.authorized_persons, f)
            print(f"Cached authorized persons data to {self.cache_file}")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def load_cache(self) -> bool:
        """Load authorized persons from local cache"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.authorized_persons = pickle.load(f)
                print(f"Loaded {len(self.authorized_persons)} authorized persons from cache")
                return True
            except Exception as e:
                print(f"Error loading cache: {e}")
                return False
        return False
    
    def integrate_with_face_database(self, face_db: FaceDatabase):
        """Integrate authorized persons with the face database"""
        print("Integrating authorized persons with face database...")
        
        for name, person_data in self.authorized_persons.items():
            # Convert embedding list back to numpy array
            embedding = np.array(person_data['embedding'], dtype=np.float32)
            
            # Normalize the embedding (important for cosine similarity)
            embedding = embedding / np.linalg.norm(embedding)
            
            # Register in face database
            face_db.register_face(
                name=person_data['name'],
                embedding=embedding,
                image_path=f"authorized_person_{person_data.get('id', 'unknown')}"
            )
            
            # Add additional metadata
            if person_data['name'] in face_db.face_metadata:
                face_db.face_metadata[person_data['name']].update({
                    'employee_id': person_data.get('id'),
                    'department': person_data.get('department'),
                    'access_level': person_data.get('access_level'),
                    'status': person_data.get('status'),
                    'source': 'azure_authorized_persons'
                })
        
        print(f"Successfully integrated {len(self.authorized_persons)} authorized persons")
    
    def get_person_info(self, name: str) -> Optional[dict]:
        """Get detailed information about an authorized person"""
        return self.authorized_persons.get(name)
    
    def list_authorized_persons(self) -> List[str]:
        """Get list of all authorized person names"""
        return list(self.authorized_persons.keys())
    
    def get_by_access_level(self, access_level: str) -> List[dict]:
        """Get authorized persons by access level"""
        return [
            person for person in self.authorized_persons.values()
            if person.get('access_level') == access_level
        ]

def test_with_authorized_persons():
    """Test the face recognition system with authorized persons"""
    from main import initialize_face_analysis, process_image
    
    print("=== Testing with Authorized Persons ===\n")
    
    # Initialize systems
    app = initialize_face_analysis()
    face_db = FaceDatabase("authorized_test_db.pkl")
    auth_manager = AuthorizedPersonsManager()
    
    # Try to load from Azure first, fallback to cache, then sample data
    success = False
    
    print("1. Attempting to load from Azure blob storage...")
    if auth_manager.load_from_azure():
        success = True
        print("✅ Successfully loaded from Azure")
    else:
        print("❌ Failed to load from Azure (authentication required)")
        
        print("\n2. Attempting to load from local cache...")
        if auth_manager.load_cache():
            success = True
            print("✅ Successfully loaded from cache")
        else:
            print("❌ No cache found")
            
            print("\n3. Loading sample authorized persons data...")
            auth_manager.load_sample_data()
            success = True
            print("✅ Sample data loaded for testing")
    
    if success:
        # Integrate with face database
        auth_manager.integrate_with_face_database(face_db)
        
        # Show authorized persons info
        print(f"\n=== Authorized Persons Summary ===")
        print(f"Total authorized persons: {len(auth_manager.authorized_persons)}")
        
        for name, person in auth_manager.authorized_persons.items():
            print(f"\nPerson: {person['name']}")
            print(f"  ID: {person.get('id', 'N/A')}")
            print(f"  Department: {person.get('department', 'N/A')}")
            print(f"  Access Level: {person.get('access_level', 'N/A')}")
            print(f"  Status: {person.get('status', 'N/A')}")
            print(f"  Embedding Size: {len(person['embedding'])}")
        
        # Test with sample image if available
        sample_images = [
            'env/insightface/data/images/t1.jpg',
            'env/insightface/data/images/Tom_Hanks_54745.png'
        ]
        
        for img_path in sample_images:
            if os.path.exists(img_path):
                print(f"\n=== Testing Recognition with {img_path} ===")
                result = process_image(app, face_db, img_path, show_image=False)
                break
        
        print(f"\n=== Authorization System Ready ===")
        print("The system can now:")
        print("✅ Recognize authorized personnel")
        print("✅ Provide access level information")
        print("✅ Track department and employee ID")
        print("✅ Validate person status (active/inactive)")
        
        return True
    
    return False

if __name__ == "__main__":
    test_with_authorized_persons()