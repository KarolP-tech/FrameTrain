#!/usr/bin/env python3
"""
Check which packages are installed and which interpreter Python is using
"""

import sys
import importlib.util
from pathlib import Path

print("=" * 60)
print("Python Environment Check")
print("=" * 60)
print()

print(f"✓ Python Executable: {sys.executable}")
print(f"✓ Python Version: {sys.version}")
print(f"✓ Python Path: {sys.prefix}")
print()

print("=" * 60)
print("Checking Required Packages")
print("=" * 60)
print()

packages = {
    # Package name: Import name
    'torch': 'torch',
    'torchvision': 'torchvision',
    'torchaudio': 'torchaudio',
    'transformers': 'transformers',
    'datasets': 'datasets',
    'timm': 'timm',
    'pillow': 'PIL',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scikit-learn': 'sklearn',
    'matplotlib': 'matplotlib',
    'opencv-python': 'cv2',
    'librosa': 'librosa',
    'soundfile': 'soundfile',
    'ultralytics': 'ultralytics',
}

installed = []
missing = []

for package_name, import_name in packages.items():
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            # Try to import to get version
            try:
                module = importlib.import_module(import_name)
                version = getattr(module, '__version__', 'unknown')
                print(f"✓ {package_name:20} -> {import_name:15} (v{version})")
                installed.append(package_name)
            except Exception as e:
                print(f"⚠ {package_name:20} -> {import_name:15} (found but error: {e})")
                installed.append(package_name)
        else:
            print(f"✗ {package_name:20} -> {import_name:15} (NOT FOUND)")
            missing.append(package_name)
    except Exception as e:
        print(f"✗ {package_name:20} -> {import_name:15} (ERROR: {e})")
        missing.append(package_name)

print()
print("=" * 60)
print("Summary")
print("=" * 60)
print(f"✓ Installed: {len(installed)}/{len(packages)}")
print(f"✗ Missing: {len(missing)}/{len(packages)}")

if missing:
    print()
    print("Missing packages:")
    for pkg in missing:
        print(f"  - {pkg}")
    print()
    print("To install missing packages:")
    print(f"  {sys.executable} -m pip install {' '.join(missing)}")
else:
    print()
    print("🎉 All packages are installed!")

print()
print("=" * 60)
print("VS Code Configuration")
print("=" * 60)
print()
print("If you see yellow import warnings in VS Code:")
print("1. Open Command Palette (Cmd+Shift+P)")
print("2. Type: 'Python: Select Interpreter'")
print("3. Select this Python:")
print(f"   {sys.executable}")
print()
