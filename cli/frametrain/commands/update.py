"""
Update command for FrameTrain CLI
"""

import os
import sys
import platform
from pathlib import Path

from ..utils import (
    print_success,
    print_error,
    print_info,
    print_warning,
    get_config,
    save_config,
    make_api_request,
    download_file,
    create_spinner
)


def check_for_updates() -> dict:
    """Check if there's a new version available"""
    try:
        response = make_api_request('/version/latest', method='GET')
        
        if response:
            return {
                'available': True,
                'version': response.get('version', 'unknown'),
                'url': response.get('download_url'),
                'changelog': response.get('changelog', [])
            }
        return {'available': False}
    except Exception as e:
        print_warning(f"⚠ Could not check for updates: {str(e)}")
        return {'available': False}


def update_app(force: bool = False):
    """Update the FrameTrain desktop application"""
    
    config = get_config()
    
    if not config:
        print_error("FrameTrain is not installed")
        sys.exit(1)
    
    current_version = config.get('version', '0.0.0')
    print_info(f"Current version: {current_version}")
    
    # Check for updates
    print_info("Checking for updates...")
    update_info = check_for_updates()
    
    if not update_info.get('available') and not force:
        print_success("✓ You're already running the latest version!")
        return
    
    new_version = update_info.get('version', 'unknown')
    
    if not force:
        print_info(f"New version available: {new_version}")
        
        # Show changelog if available
        changelog = update_info.get('changelog', [])
        if changelog:
            print_info("\nWhat's new:")
            for change in changelog:
                print_info(f"  • {change}")
        
        if not click.confirm('\nDo you want to update?', default=True):
            return
    
    # Download new version
    app_path = Path(config.get('app_path', ''))
    backup_path = app_path.with_suffix('.backup')
    
    try:
        # Backup current version
        if app_path.exists():
            print_info("Creating backup...")
            import shutil
            shutil.copy2(app_path, backup_path)
        
        # Download new version
        os_type = platform.system()
        download_url = update_info.get('url') or get_download_url(os_type)
        
        print_info("Downloading update...")
        with create_spinner("Downloading..."):
            download_file(download_url, app_path)
        
        # Make executable on Unix
        if os_type in ["Darwin", "Linux"]:
            os.chmod(app_path, 0o755)
        
        # Update config
        config['version'] = new_version
        save_config(config)
        
        # Remove backup
        if backup_path.exists():
            backup_path.unlink()
        
        print_success("\n✓ Update completed successfully! 🎉")
        print_info(f"Updated to version {new_version}")
        
    except Exception as e:
        print_error(f"✗ Update failed: {str(e)}")
        
        # Restore backup
        if backup_path.exists():
            print_info("Restoring backup...")
            import shutil
            shutil.copy2(backup_path, app_path)
            backup_path.unlink()
            print_success("✓ Backup restored")
        
        sys.exit(1)


def get_download_url(os_type: str) -> str:
    """Get the download URL for the appropriate OS"""
    base_url = os.getenv("FRAMETRAIN_DOWNLOAD_URL", "https://downloads.frametrain.ai")
    
    urls = {
        "Windows": f"{base_url}/frametrain-latest-windows.exe",
        "Darwin": f"{base_url}/frametrain-latest-macos.dmg",
        "Linux": f"{base_url}/frametrain-latest-linux.AppImage",
    }
    
    return urls.get(os_type, urls["Linux"])
