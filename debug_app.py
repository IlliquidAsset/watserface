#!/usr/bin/env python3
"""
Enhanced debugging version of app.py with comprehensive logging
"""

import sys
import subprocess
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_app.log')
    ]
)

logger = logging.getLogger('FACEFUSION_DEBUG')

def log_system_info():
    """Log system information for debugging"""
    try:
        import platform
        import psutil
        
        logger.info("=== SYSTEM INFORMATION ===")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Python: {sys.version}")
        logger.info(f"CPU: {psutil.cpu_count()} cores")
        logger.info(f"Memory: {psutil.virtual_memory().total // (1024**3)} GB")
        logger.info(f"Working Directory: {sys.path[0]}")
        logger.info("==========================")
    except ImportError:
        logger.warning("psutil not available, skipping system info")

def install_with_logging(packages, description):
    """Install packages with detailed logging"""
    logger.info(f"🔧 Installing {description}...")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir"] + packages,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {description} installed successfully")
            if result.stdout:
                logger.debug(f"Install output: {result.stdout}")
        else:
            logger.error(f"❌ {description} installation failed")
            logger.error(f"Error output: {result.stderr}")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ {description} installation timed out")
        return False
    except Exception as e:
        logger.error(f"💥 Exception during {description} installation: {e}")
        traceback.print_exc()
        return False

def main():
    """Enhanced main function with comprehensive logging"""
    logger.info("🚀 Starting FaceFusion with Enhanced Debugging")
    log_system_info()
    
    try:
        # 1) Install ONNX runtime
        logger.info("📦 Step 1: Installing ONNX Runtime GPU")
        onnx_success = install_with_logging(
            ["onnxruntime-gpu"], 
            "ONNX Runtime GPU"
        )
        
        if not onnx_success:
            logger.warning("⚠️ ONNX GPU installation failed, continuing anyway...")

        # 2) Install FaceFusion dependencies
        logger.info("📦 Step 2: Installing FaceFusion Dependencies")
        try:
            result = subprocess.run([
                sys.executable, "install.py",
                "--onnxruntime", "cuda",
                "--skip-conda"
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("✅ FaceFusion dependencies installed")
                if result.stdout:
                    logger.debug(f"Install output: {result.stdout}")
            else:
                logger.error("❌ FaceFusion dependency installation failed")
                logger.error(f"Error: {result.stderr}")
                
        except Exception as e:
            logger.error(f"💥 Exception during FaceFusion installation: {e}")
            traceback.print_exc()

        # 3) Install training dependencies
        logger.info("📦 Step 3: Installing Training Dependencies")
        training_packages = [
            "torch==2.1.0+cpu", 
            "torchvision==0.16.0+cpu",
            "--extra-index-url", "https://download.pytorch.org/whl/cpu",
            "transformers>=4.30.0",
            "diffusers>=0.20.0",
            "huggingface_hub>=0.16.4", 
            "matplotlib>=3.7.0",
            "accelerate>=0.20.3"
        ]
        
        training_success = install_with_logging(
            training_packages,
            "Training Dependencies (PyTorch CPU + ML libs)"
        )
        
        if training_success:
            logger.info("🎯 Training functionality will be available!")
        else:
            logger.warning("⚠️ Training dependencies failed, training will be limited")

        # 4) Test imports before launching
        logger.info("🧪 Step 4: Testing Critical Imports")
        try:
            import gradio
            logger.info(f"✅ Gradio {gradio.__version__} imported")
            
            from facefusion import core
            logger.info("✅ FaceFusion core imported")
            
            from facefusion.uis.layouts import default, training
            logger.info("✅ UI layouts imported")
            
            # Test torch import
            try:
                import torch
                logger.info(f"✅ PyTorch {torch.__version__} imported")
            except ImportError:
                logger.warning("⚠️ PyTorch not available - training will be limited")
                
        except Exception as e:
            logger.error(f"💥 Critical import failed: {e}")
            traceback.print_exc()

        # 5) Launch the app
        logger.info("🚀 Step 5: Launching FaceFusion with Training Tab")
        try:
            launch_command = [
                sys.executable, "facefusion.py", "run",
                "--execution-providers", "cuda",
                "--ui-layouts", "default", "training"
            ]
            
            logger.info(f"Launch command: {' '.join(launch_command)}")
            
            # Launch without capturing output so we can see real-time logs
            process = subprocess.Popen(launch_command)
            logger.info(f"✅ FaceFusion launched with PID: {process.pid}")
            
            # Wait for the process
            process.wait()
            
            if process.returncode == 0:
                logger.info("✅ FaceFusion exited cleanly")
            else:
                logger.error(f"❌ FaceFusion exited with code: {process.returncode}")
                
        except Exception as e:
            logger.error(f"💥 Exception during launch: {e}")
            traceback.print_exc()

    except Exception as e:
        logger.error(f"💥 Critical error in main: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()