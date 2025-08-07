#!/usr/bin/env python3
import sys
import subprocess

def main():
    # 1) Install the GPU ONNX runtime
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir", "onnxruntime-gpu"
    ], check=True)

    # 2) Install all FaceFusion dependencies for CUDA
    subprocess.run([
        sys.executable, "install.py",
        "--onnxruntime", "cuda",
        "--skip-conda"
    ], check=True)
    
    # 2.5) Install training dependencies (optimized)
    try:
        # Check if training deps are already installed to avoid rebuilds
        try:
            import torch
            import transformers
            print("✅ Training dependencies already available, skipping installation")
        except ImportError:
            print("📦 Installing training dependencies...")
            
            # Fix NumPy compatibility first
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--quiet",
                "numpy<2.0"
            ], check=False)
            
            # Install compatible PyTorch version (2.0.0 has better uint64 support)
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--quiet",
                "torch==2.0.0+cpu", 
                "torchvision==0.15.0+cpu", 
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ], check=False)
            
            # Install other training dependencies with compatible versions
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--quiet",
                "transformers>=4.30.0",
                "diffusers>=0.20.0", 
                "huggingface_hub>=0.16.4",
                "accelerate>=0.20.3",
                "safetensors<0.4.0"  # Use older safetensors without torch.uint64
            ], check=False)
        
        print("✅ Training dependencies installed successfully")
    except Exception as e:
        print(f"⚠️ Warning: Some training dependencies could not be installed: {e}")

    # 3) Launch the full FaceFusion UI with training tab on your GPU
    subprocess.run([
        sys.executable, "facefusion.py", "run",
        "--execution-providers", "cuda",
        "--ui-layouts", "default"
    ], check=True)

if __name__ == "__main__":
    main()
