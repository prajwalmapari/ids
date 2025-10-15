#!/usr/bin/env python3
"""
Debug CUDA DLL loading
"""
import os
import sys

# Get site-packages directory
site_packages = None
for path in sys.path:
    if path.endswith('site-packages'):
        site_packages = path
        break

print(f"Site packages: {site_packages}")

if site_packages:
    cuda_paths = [
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
    
    print("\n🔍 Checking CUDA paths:")
    for cuda_path in cuda_paths:
        exists = os.path.exists(cuda_path)
        if exists:
            files = os.listdir(cuda_path)
            dll_files = [f for f in files if f.endswith('.dll')]
            print(f"✅ {cuda_path} - {len(dll_files)} DLL files")
            if len(dll_files) > 0:
                print(f"   📄 Sample DLLs: {dll_files[:3]}")
        else:
            print(f"❌ {cuda_path} - NOT FOUND")
    
    # Add DLL directories
    print("\n🔧 Adding DLL directories:")
    if hasattr(os, 'add_dll_directory'):
        for cuda_path in cuda_paths:
            if os.path.exists(cuda_path):
                try:
                    os.add_dll_directory(cuda_path)
                    print(f"✅ Added: {cuda_path}")
                except Exception as e:
                    print(f"❌ Failed: {cuda_path} - {e}")
    
    # Test ONNX Runtime
    print("\n🧪 Testing ONNX Runtime CUDA:")
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"✅ Available providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            # Try to create a simple session
            try:
                session_options = ort.SessionOptions()
                session_options.providers = ['CUDAExecutionProvider']
                print("✅ CUDA provider can be set in session options")
            except Exception as e:
                print(f"❌ Failed to set CUDA provider: {e}")
    except Exception as e:
        print(f"❌ ONNX Runtime error: {e}")

print("\n🏁 Debug completed")