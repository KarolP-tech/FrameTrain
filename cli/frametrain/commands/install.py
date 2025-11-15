"""
Installation command for FrameTrain CLI
"""

import os
import sys
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from ..utils import (
    print_success,
    print_error,
    print_info,
    print_warning,
    make_api_request,
    download_file,
    get_config,
    save_config,
    create_spinner
)


def get_default_install_path() -> Path:
    """Get the default installation path based on OS"""
    if platform.system() == "Windows":
        return Path.home() / "AppData" / "Local" / "FrameTrain"
    elif platform.system() == "Darwin":  # macOS
        return Path.home() / "Applications" / "FrameTrain"
    else:  # Linux
        return Path.home() / ".local" / "share" / "frametrain"


def get_platform_identifier() -> str:
    """Get the platform identifier for API requests"""
    system = platform.system()
    if system == "Windows":
        return "windows"
    elif system == "Darwin":
        return "mac"
    else:
        return "linux"


def verify_key_with_server(api_key: str) -> bool:
    """Verify the API key with the FrameTrain server"""
    try:
        response = make_api_request(
            '/keys/verify',
            method='POST',
            data={'key': api_key}
        )
        
        if response and response.get('valid'):
            print_success("✓ API key verified successfully")
            return True
        else:
            print_error("✗ Invalid API key")
            return False
    except Exception as e:
        print_error(f"✗ Failed to verify key: {str(e)}")
        return False


def get_download_info(api_key: str, platform_id: str) -> dict:
    """Get download information from the API"""
    try:
        response = make_api_request(
            f'/download-app?platform={platform_id}&key={api_key}',
            method='GET'
        )
        
        if not response or not response.get('success'):
            raise Exception(response.get('message', 'Failed to get download info'))
        
        return response
    except Exception as e:
        raise Exception(f"Failed to fetch download info: {str(e)}")


def install_windows(download_url: str, install_path: Path) -> Path:
    """Install on Windows"""
    print_info("Downloading Windows installer...")
    
    installer_path = install_path / "FrameTrain-Setup.msi"
    download_file(download_url, installer_path)
    
    print_info("Running Windows installer...")
    print_warning("Please follow the installation wizard...")
    
    # Run MSI installer
    subprocess.run(['msiexec', '/i', str(installer_path)], check=True)
    
    # Try to find installed app
    possible_paths = [
        Path(os.environ.get('ProgramFiles', 'C:\\Program Files')) / "FrameTrain" / "FrameTrain.exe",
        Path(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)')) / "FrameTrain" / "FrameTrain.exe",
        Path(os.environ.get('LOCALAPPDATA', '')) / "Programs" / "FrameTrain" / "FrameTrain.exe"
    ]
    
    for app_path in possible_paths:
        if app_path.exists():
            return app_path
    
    print_warning("Could not automatically detect installation path")
    return installer_path


def install_macos(download_url: str, install_path: Path) -> Path:
    """Install on macOS"""
    print_info("Downloading macOS installer...")
    
    dmg_path = install_path / "FrameTrain.dmg"
    download_file(download_url, dmg_path)
    
    print_info("Mounting DMG...")
    mount_result = subprocess.run(
        ['hdiutil', 'attach', str(dmg_path), '-nobrowse'],
        capture_output=True,
        text=True
    )
    
    if mount_result.returncode != 0:
        raise Exception("Failed to mount DMG")
    
    # Find mount point
    mount_point = None
    for line in mount_result.stdout.split('\n'):
        if '/Volumes/' in line:
            mount_point = line.split('\t')[-1].strip()
            break
    
    if not mount_point:
        raise Exception("Could not find mount point")
    
    print_info(f"Copying app to {install_path}...")
    
    # Find .app bundle in mount
    app_name = None
    mount_path = Path(mount_point)
    for item in mount_path.iterdir():
        if item.suffix == '.app':
            app_name = item.name
            break
    
    if not app_name:
        subprocess.run(['hdiutil', 'detach', mount_point])
        raise Exception("Could not find .app bundle in DMG")
    
    # Copy to Applications
    app_source = mount_path / app_name
    app_dest = install_path / app_name
    
    if app_dest.exists():
        shutil.rmtree(app_dest)
    
    shutil.copytree(app_source, app_dest)
    
    # Unmount DMG
    subprocess.run(['hdiutil', 'detach', mount_point])
    
    print_success(f"✓ App installed to {app_dest}")
    return app_dest


def install_linux(download_url: str, install_path: Path) -> Path:
    """Install on Linux"""
    print_info("Downloading Linux AppImage...")
    
    appimage_path = install_path / "FrameTrain.AppImage"
    download_file(download_url, appimage_path)
    
    print_info("Making AppImage executable...")
    os.chmod(appimage_path, 0o755)
    
    # Create desktop entry
    try:
        create_linux_desktop_entry(appimage_path)
        print_success("✓ Desktop entry created")
    except Exception as e:
        print_warning(f"⚠ Could not create desktop entry: {str(e)}")
    
    return appimage_path


def create_linux_desktop_entry(app_path: Path):
    """Create a .desktop file for Linux"""
    desktop_file = Path.home() / ".local" / "share" / "applications" / "frametrain.desktop"
    desktop_file.parent.mkdir(parents=True, exist_ok=True)
    
    desktop_content = f"""[Desktop Entry]
Type=Application
Name=FrameTrain
Comment=Local ML Training Platform
Exec={app_path}
Icon=frametrain
Terminal=false
Categories=Development;Science;Education;
"""
    desktop_file.write_text(desktop_content)


def install_app(api_key: str, install_path: Optional[str] = None):
    """Install the FrameTrain desktop application"""
    
    # Verify API key first
    print_info("Verifying API key...")
    if not verify_key_with_server(api_key):
        sys.exit(1)
    
    # Determine installation path
    if install_path:
        target_path = Path(install_path)
    else:
        target_path = get_default_install_path()
    
    print_info(f"Installation path: {target_path}")
    
    # Create installation directory
    try:
        target_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print_error(f"Failed to create installation directory: {str(e)}")
        sys.exit(1)
    
    # Get platform
    os_type = platform.system()
    platform_id = get_platform_identifier()
    
    print_info(f"Detected platform: {os_type}")
    
    # Get download info from API
    try:
        print_info("Fetching download information...")
        download_info = get_download_info(api_key, platform_id)
        
        download_url = download_info['download_url']
        version = download_info['version']
        size_mb = download_info['size_mb']
        
        print_info(f"Version: {version}")
        print_info(f"Size: {size_mb} MB")
        
    except Exception as e:
        print_error(f"✗ {str(e)}")
        sys.exit(1)
    
    # Download and install based on platform
    try:
        with create_spinner("Installing..."):
            if os_type == "Windows":
                app_path = install_windows(download_url, target_path)
            elif os_type == "Darwin":
                app_path = install_macos(download_url, target_path)
            else:  # Linux
                app_path = install_linux(download_url, target_path)
        
    except Exception as e:
        print_error(f"✗ Installation failed: {str(e)}")
        sys.exit(1)
    
    # Save configuration
    config = {
        'api_key': api_key,
        'install_path': str(target_path),
        'app_path': str(app_path),
        'version': version,
        'platform': platform_id,
    }
    
    save_config(config)
    print_success("✓ Configuration saved")
    
    print_success("\n" + "="*60)
    print_success("🎉 FrameTrain installed successfully!")
    print_success("="*60)
    print_info(f"\n📍 Installation location: {app_path}")
    print_info(f"📦 Version: {version}")
    print_info("\n🚀 To start FrameTrain, run:")
    print_info("   frametrain start")
    print_info("\n📚 For help and documentation:")
    print_info("   https://docs.frametrain.ai")


def uninstall_app():
    """Uninstall the FrameTrain application"""
    config = get_config()
    
    if not config:
        print_error("FrameTrain is not installed")
        return
    
    install_path = Path(config.get('install_path', ''))
    
    if not install_path.exists():
        print_error("Installation directory not found")
        return
    
    try:
        print_info(f"Removing {install_path}...")
        shutil.rmtree(install_path)
        
        # Remove config file
        config_path = Path.home() / ".frametrain" / "config.json"
        if config_path.exists():
            config_path.unlink()
        
        # Remove Linux desktop entry if exists
        if platform.system() == "Linux":
            desktop_file = Path.home() / ".local" / "share" / "applications" / "frametrain.desktop"
            if desktop_file.exists():
                desktop_file.unlink()
        
        print_success("✓ FrameTrain uninstalled successfully")
    except Exception as e:
        print_error(f"✗ Uninstallation failed: {str(e)}")
        sys.exit(1)
