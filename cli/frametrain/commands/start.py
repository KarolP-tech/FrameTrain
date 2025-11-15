"""
Start command for FrameTrain CLI
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

from ..utils import (
    print_success,
    print_error,
    print_info,
    get_config,
    make_api_request
)


def verify_key_before_start() -> bool:
    """Verify API key before starting the app"""
    config = get_config()
    
    if not config or 'api_key' not in config:
        print_error("API key not found. Please run 'frametrain install' first.")
        return False
    
    api_key = config['api_key']
    
    try:
        response = make_api_request(
            '/keys/verify',
            method='POST',
            data={'key': api_key}
        )
        
        if response and response.get('valid'):
            print_success("✓ API key verified")
            return True
        else:
            print_error("✗ API key is invalid or expired")
            print_info("Please purchase a new key at https://frametrain.ai")
            return False
    except Exception as e:
        print_error(f"✗ Failed to verify key: {str(e)}")
        return False


def start_app(verify: bool = True):
    """Start the FrameTrain desktop application"""
    
    # Verify key if requested
    if verify and not verify_key_before_start():
        sys.exit(1)
    
    # Get configuration
    config = get_config()
    
    if not config:
        print_error("FrameTrain is not installed. Run 'frametrain install' first.")
        sys.exit(1)
    
    app_path = config.get('app_path')
    
    if not app_path or not Path(app_path).exists():
        print_error("FrameTrain application not found")
        print_info("Try reinstalling: frametrain install --key YOUR_KEY")
        sys.exit(1)
    
    # Start the application
    print_info(f"Starting FrameTrain from {app_path}...")
    
    try:
        os_type = platform.system()
        
        if os_type == "Windows":
            # Windows: Start the .exe
            subprocess.Popen([app_path], shell=True)
        elif os_type == "Darwin":
            # macOS: Open the app bundle
            subprocess.Popen(['open', app_path])
        else:
            # Linux: Execute the AppImage
            subprocess.Popen([app_path])
        
        print_success("✓ FrameTrain started successfully!")
        
    except Exception as e:
        print_error(f"✗ Failed to start FrameTrain: {str(e)}")
        sys.exit(1)
