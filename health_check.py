#!/usr/bin/env python3
"""
Health check script for FaceFusion Space
"""
import requests
import time
import sys

def check_space_health():
    space_url = "https://IlliquidAsset-facefusion3.1.hf.space"
    space_api = "https://huggingface.co/api/spaces/IlliquidAsset/facefusion3.1"
    
    print("🏥 FaceFusion Space Health Check")
    print("================================")
    
    # Check Space API status
    try:
        print(f"📡 Checking Space API status...")
        response = requests.get(space_api, timeout=10)
        if response.status_code == 200:
            data = response.json()
            status = data.get('runtime', {}).get('stage', 'unknown')
            print(f"✅ Space API Status: {status}")
            
            if status == 'RUNNING':
                print("🟢 Space is running!")
            elif status == 'BUILDING':
                print("🟡 Space is building...")
            elif status == 'STOPPED':
                print("🔴 Space is stopped")
            else:
                print(f"🟠 Space status: {status}")
        else:
            print(f"❌ Space API error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Space API check failed: {e}")
    
    # Check runtime endpoint
    try:
        print(f"🌐 Checking runtime endpoint...")
        response = requests.get(space_url, timeout=15)
        if response.status_code == 200:
            print("✅ Runtime endpoint responding")
            if "gradio" in response.text.lower():
                print("✅ Gradio interface detected")
            else:
                print("⚠️ No Gradio interface detected")
        else:
            print(f"❌ Runtime endpoint error: {response.status_code}")
            
    except requests.exceptions.Timeout:
        print("⏰ Runtime endpoint timeout (Space may be starting)")
    except Exception as e:
        print(f"❌ Runtime check failed: {e}")

if __name__ == "__main__":
    check_space_health()