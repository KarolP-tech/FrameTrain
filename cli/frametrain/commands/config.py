"""
Config command for FrameTrain CLI
"""

import sys
from pathlib import Path
from typing import Optional

from ..utils import (
    print_success,
    print_error,
    print_info,
    get_config,
    save_config
)


def manage_config(action: str, key: Optional[str] = None, url: Optional[str] = None):
    """Manage FrameTrain configuration"""
    
    if action == 'show':
        show_config()
    elif action == 'set-key':
        if not key:
            print_error("Please provide a key using --key option")
            sys.exit(1)
        set_api_key(key)
    elif action == 'set-url':
        if not url:
            print_error("Please provide a URL using --url option")
            sys.exit(1)
        set_api_url(url)


def show_config():
    """Display current configuration"""
    config = get_config()
    
    if not config:
        print_error("No configuration found")
        print_info("Run 'frametrain install' to set up FrameTrain")
        return
    
    print_info("\n" + "="*50)
    print_info("FrameTrain Configuration")
    print_info("="*50)
    
    # Mask API key for security
    api_key = config.get('api_key', 'Not set')
    if api_key and api_key != 'Not set':
        masked_key = api_key[:8] + '...' + api_key[-4:] if len(api_key) > 12 else api_key
    else:
        masked_key = 'Not set'
    
    print_info(f"API Key: {masked_key}")
    print_info(f"Install Path: {config.get('install_path', 'Not set')}")
    print_info(f"App Path: {config.get('app_path', 'Not set')}")
    print_info(f"Version: {config.get('version', 'Unknown')}")
    print_info(f"API URL: {config.get('api_url', 'Default')}")
    print_info("="*50 + "\n")


def set_api_key(api_key: str):
    """Set or update the API key"""
    config = get_config() or {}
    
    config['api_key'] = api_key
    save_config(config)
    
    print_success("✓ API key updated successfully")


def set_api_url(api_url: str):
    """Set or update the API URL"""
    config = get_config() or {}
    
    config['api_url'] = api_url
    save_config(config)
    
    print_success(f"✓ API URL set to: {api_url}")


def show_info():
    """Show detailed FrameTrain information"""
    import platform
    
    print_info("\n" + "="*50)
    print_info("FrameTrain System Information")
    print_info("="*50)
    
    print_info(f"Operating System: {platform.system()} {platform.release()}")
    print_info(f"Python Version: {platform.python_version()}")
    print_info(f"Architecture: {platform.machine()}")
    
    config = get_config()
    if config:
        print_info(f"\nInstallation Status: ✓ Installed")
        print_info(f"Version: {config.get('version', 'Unknown')}")
        print_info(f"Install Location: {config.get('install_path', 'Unknown')}")
        
        app_path = Path(config.get('app_path', ''))
        if app_path.exists():
            size = app_path.stat().st_size / (1024 * 1024)  # MB
            print_info(f"App Size: {size:.2f} MB")
        
        # Check if key is set
        if config.get('api_key'):
            print_info("API Key: ✓ Configured")
        else:
            print_info("API Key: ✗ Not configured")
    else:
        print_info("\nInstallation Status: ✗ Not installed")
    
    print_info("="*50 + "\n")
    print_info("For more information, visit: https://frametrain.ai")
    print_info("Support: support@frametrain.ai")
