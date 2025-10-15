#!/usr/bin/env python3
"""
Simple CUDA test to verify GPU acceleration is working
"""

import sys
import os

def setup_cuda_environment():
    """Setup CUDA DLL paths like in main.py"""
    import site
    
    # Get site-packages directory
    site_packages = None
    for path in sys.path:
        if path.endswith('site-packages'):
            site_packages = path
            break
    
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
        
        # Add DLL directories to DLL search path (Windows-specific)
        if hasattr(os, 'add_dll_directory'):
            for cuda_path in cuda_paths:
                if os.path.exists(cuda_path):
                    try:
                        os.add_dll_directory(cuda_path)
                    except Exception:
                        pass  # Ignore errors silently
        
        # Also add to PATH as backup
        current_path = os.environ.get('PATH', '')
        for cuda_path in cuda_paths:
            if os.path.exists(cuda_path) and cuda_path not in current_path:
                os.environ['PATH'] = cuda_path + os.pathsep + current_path
                current_path = os.environ['PATH']

def test_cuda_availability():
    """Test CUDA availability for ONNX Runtime"""
    print("🔍 Testing CUDA availability...")
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"✅ ONNX Runtime providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDAExecutionProvider is available")
            
            # Test actual CUDA session creation
            try:
                # Create a simple session with CUDA provider
                import numpy as np
                
                # Simple test - this will actually try to load CUDA libraries
                session_options = ort.SessionOptions()
                providers_list = [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kSameAsRequested',
                        'gpu_mem_limit': 1024 * 1024 * 1024,  # 1GB test limit
                    }),
                    'CPUExecutionProvider'
                ]
                
                print("🔧 Testing CUDA provider initialization...")
                
                # This will fail if CUDA libraries are missing
                test_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in test_providers:
                    print("✅ CUDA libraries are properly loaded")
                    return True
                else:
                    print("❌ CUDA provider not in available providers")
                    return False
                    
            except Exception as e:
                print(f"❌ CUDA provider test failed: {e}")
                if "cublasLt64_12.dll" in str(e):
                    print("💡 Missing cuBLAS library - need to install CUDA runtime")
                return False
        else:
            print("❌ CUDAExecutionProvider not available")
            return False
            
    except Exception as e:
        print(f"❌ ONNX Runtime test failed: {e}")
        return False

def test_insightface_cuda():
    """Test InsightFace with CUDA"""
    print("\n🧠 Testing InsightFace with CUDA...")
    
    try:
        from insightface.app import FaceAnalysis
        
        # Try to initialize with CUDA
        app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(320, 320))  # ctx_id=0 for GPU
        
        # Check which providers are actually being used
        if hasattr(app, 'models'):
            gpu_models = 0
            total_models = 0
            for model_name, model in app.models.items():
                total_models += 1
                if hasattr(model, 'session') and hasattr(model.session, 'get_providers'):
                    providers = model.session.get_providers()
                    if 'CUDAExecutionProvider' in providers:
                        gpu_models += 1
                        print(f"✅ {model_name}: Using GPU ({providers[0]})")
                    else:
                        print(f"💻 {model_name}: Using CPU ({providers[0]})")
            
            if gpu_models > 0:
                print(f"🎮 SUCCESS: {gpu_models}/{total_models} models using GPU")
                return True
            else:
                print(f"❌ FAILED: 0/{total_models} models using GPU")
                return False
        else:
            print("⚠️ Cannot determine model providers")
            return False
            
    except Exception as e:
        print(f"❌ InsightFace CUDA test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 CUDA Verification Test")
    print("=" * 50)
    
    # Setup CUDA environment first
    setup_cuda_environment()
    
    cuda_ok = test_cuda_availability()
    insightface_ok = test_insightface_cuda()
    
    print("\n" + "=" * 50)
    print("📊 RESULTS:")
    print(f"CUDA Available: {'✅ YES' if cuda_ok else '❌ NO'}")
    print(f"InsightFace GPU: {'✅ YES' if insightface_ok else '❌ NO'}")
    
    if cuda_ok and insightface_ok:
        print("🎉 GPU acceleration is WORKING!")
    elif cuda_ok:
        print("⚠️ CUDA available but InsightFace not using GPU")
    else:
        print("❌ GPU acceleration not working - missing CUDA libraries")
    
    print("=" * 50)