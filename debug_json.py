#!/usr/bin/env python3
"""
Debug script to examine the actual JSON structure from Azure
"""

from azure_integration import AuthorizedPersonsManager
import json

def debug_azure_json():
    """Debug the actual JSON structure"""
    print("=== DEBUGGING AZURE JSON STRUCTURE ===\n")
    
    auth_manager = AuthorizedPersonsManager()
    
    # Try to load from Azure
    try:
        from azure.storage.blob import BlobServiceClient
        
        print("Connecting to Azure Storage...")
        blob_service_client = BlobServiceClient.from_connection_string(auth_manager.connection_string)
        
        # Get blob client
        blob_client = blob_service_client.get_blob_client(
            container=auth_manager.container_name, 
            blob=auth_manager.blob_name
        )
        
        # Download blob content
        blob_data = blob_client.download_blob()
        content = blob_data.readall()
        
        # Parse JSON
        raw_data = json.loads(content.decode('utf-8'))
        
        print("Raw JSON structure:")
        print("=" * 50)
        
        # Show the structure
        print(f"Data type: {type(raw_data)}")
        
        if isinstance(raw_data, dict):
            print(f"Top-level keys: {list(raw_data.keys())}")
            
            # Show first few entries
            count = 0
            for key, value in raw_data.items():
                if count >= 3:  # Show only first 3 entries
                    print("... (showing only first 3 entries)")
                    break
                    
                print(f"\nEntry {count + 1}: {key}")
                print(f"  Type: {type(value)}")
                
                if isinstance(value, dict):
                    print(f"  Keys: {list(value.keys())}")
                    
                    # Look for name fields
                    potential_name_fields = ['name', 'full_name', 'display_name', 'person_name', 'employee_name']
                    for field in potential_name_fields:
                        if field in value:
                            print(f"  🎯 Found name field '{field}': {value[field]}")
                    
                    # Show first few fields of the entry
                    shown_fields = 0
                    for subkey, subvalue in value.items():
                        if shown_fields >= 5:
                            print("    ... (more fields)")
                            break
                        if subkey != 'embedding':  # Skip embedding as it's too long
                            print(f"    {subkey}: {subvalue}")
                        else:
                            print(f"    {subkey}: [array of {len(subvalue)} values]")
                        shown_fields += 1
                
                count += 1
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        print(f"Error debugging JSON: {e}")

if __name__ == "__main__":
    debug_azure_json()