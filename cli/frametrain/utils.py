"""
Utility functions for FrameTrain CLI
"""

import os
import sys
import json
import requests
import time
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import contextmanager
from colorama import Fore, Style


# ============================================
# Output Functions
# ============================================

def print_success(message: str):
    """Print success message in green"""
    print(f"{Fore.GREEN}{message}{Style.RESET_ALL}")


def print_error(message: str):
    """Print error message in red"""
    print(f"{Fore.RED}{message}{Style.RESET_ALL}", file=sys.stderr)


def print_warning(message: str):
    """Print warning message in yellow"""
    print(f"{Fore.YELLOW}{message}{Style.RESET_ALL}")


def print_info(message: str):
    """Print info message in cyan"""
    print(f"{Fore.CYAN}{message}{Style.RESET_ALL}")


def print_banner():
    """Print FrameTrain banner"""
    banner = f"""
{Fore.BLUE}
  ______                     _______        _       
 |  ____|                   |__   __|      (_)      
 | |__ _ __ __ _ _ __ ___   ___| |_ __ __ _ _ _ __  
 |  __| '__/ _` | '_ ` _ \\ / _ \\ | '__/ _` | | '_ \\ 
 | |  | | | (_| | | | | | |  __/ | | | (_| | | | | |
 |_|  |_|  \\__,_|_| |_| |_|\\___|_|_|  \\__,_|_|_| |_|
                                                     
{Fore.CYAN}Professional Platform for Local ML Training{Style.RESET_ALL}
{Fore.YELLOW}Version 1.0.0{Style.RESET_ALL}
"""
    print(banner)


# ============================================
# Configuration Management
# ============================================

def get_config_path() -> Path:
    """Get the path to the configuration file"""
    return Path.home() / ".frametrain" / "config.json"


def get_config() -> Optional[Dict[str, Any]]:
    """Load configuration from file"""
    config_path = get_config_path()
    
    if not config_path.exists():
        return None
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print_warning(f"Failed to load config: {str(e)}")
        return None


def save_config(config: Dict[str, Any]):
    """Save configuration to file"""
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print_error(f"Failed to save config: {str(e)}")
        raise


# ============================================
# API Functions
# ============================================

def get_api_url() -> str:
    """Get the API base URL"""
    config = get_config()
    
    if config and 'api_url' in config:
        return config['api_url']
    
    return os.getenv('FRAMETRAIN_API_URL', 'https://api.frametrain.ai')


def make_api_request(
    endpoint: str,
    method: str = 'GET',
    data: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    timeout: int = 30
) -> Optional[Dict]:
    """Make an API request to the FrameTrain server"""
    
    base_url = get_api_url()
    url = f"{base_url}{endpoint}"
    
    if headers is None:
        headers = {}
    
    headers['Content-Type'] = 'application/json'
    headers['User-Agent'] = 'FrameTrain-CLI/1.0.0'
    
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers, timeout=timeout)
        elif method == 'POST':
            response = requests.post(url, json=data, headers=headers, timeout=timeout)
        elif method == 'PUT':
            response = requests.put(url, json=data, headers=headers, timeout=timeout)
        elif method == 'DELETE':
            response = requests.delete(url, headers=headers, timeout=timeout)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.Timeout:
        print_error("Request timed out")
        return None
    except requests.exceptions.ConnectionError:
        print_error("Could not connect to server")
        return None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print_error("Endpoint not found")
        elif e.response.status_code == 401:
            print_error("Unauthorized")
        elif e.response.status_code == 500:
            print_error("Server error")
        else:
            print_error(f"HTTP error: {e.response.status_code}")
        return None
    except Exception as e:
        print_error(f"Request failed: {str(e)}")
        return None


# ============================================
# Download Functions
# ============================================

def download_file(url: str, destination: Path, chunk_size: int = 8192):
    """Download a file with progress indication"""
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Show progress
                        progress = (downloaded / total_size) * 100
                        bar_length = 40
                        filled = int(bar_length * downloaded / total_size)
                        bar = '█' * filled + '-' * (bar_length - filled)
                        print(f'\r[{bar}] {progress:.1f}%', end='', flush=True)
                
                print()  # New line after progress bar
        
        return True
    
    except Exception as e:
        print_error(f"Download failed: {str(e)}")
        if destination.exists():
            destination.unlink()
        raise


# ============================================
# Spinner Context Manager
# ============================================

@contextmanager
def create_spinner(message: str = "Loading..."):
    """Create a simple spinner for long-running operations"""
    
    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    
    import threading
    
    stop_spinner = threading.Event()
    
    def spin():
        idx = 0
        while not stop_spinner.is_set():
            sys.stdout.write(f'\r{Fore.CYAN}{spinner_chars[idx]} {message}{Style.RESET_ALL}')
            sys.stdout.flush()
            idx = (idx + 1) % len(spinner_chars)
            time.sleep(0.1)
        sys.stdout.write('\r' + ' ' * (len(message) + 10) + '\r')
        sys.stdout.flush()
    
    spinner_thread = threading.Thread(target=spin)
    spinner_thread.start()
    
    try:
        yield
    finally:
        stop_spinner.set()
        spinner_thread.join()


# ============================================
# Date Formatting
# ============================================

def format_date(date_str: str) -> str:
    """Format an ISO date string to a readable format"""
    from datetime import datetime
    
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return date_str


# ============================================
# File Size Formatting
# ============================================

def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"
