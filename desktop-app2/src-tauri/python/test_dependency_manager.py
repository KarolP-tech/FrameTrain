#!/usr/bin/env python3
"""
Test Script for Plugin Dependency Manager
==========================================

Demonstrates:
1. Checking plugin dependencies
2. Auto-installing missing packages
3. Loading plugins with dependency management
4. Handling missing dependencies gracefully
"""

import sys
from pathlib import Path

# Add proto directory to path
proto_dir = Path(__file__).parent
sys.path.insert(0, str(proto_dir))

from plugin_dependency_manager import (
    DependencyManager,
    PluginLoader,
    PluginManifest
)


def test_1_parse_manifest():
    """Test: Parse plugin manifest"""
    print("\n" + "="*70)
    print("TEST 1: Parse Plugin Manifest")
    print("="*70)
    
    plugin_path = proto_dir / "plugin_vision_test.py"
    manifest = PluginManifest.from_docstring(plugin_path)
    
    if manifest:
        print(f"✓ Successfully parsed manifest")
        print(f"  Name: {manifest.plugin_name}")
        print(f"  Description: {manifest.description}")
        print(f"  Modality: {manifest.modality}")
        print(f"  Required packages: {manifest.required_packages}")
        print(f"  Optional packages: {manifest.optional_packages}")
        print(f"  Min Python: {manifest.min_python_version}")
    else:
        print("✗ Failed to parse manifest")


def test_2_check_dependencies():
    """Test: Check which dependencies are installed"""
    print("\n" + "="*70)
    print("TEST 2: Check Dependencies")
    print("="*70)
    
    dep_manager = DependencyManager()
    plugin_path = proto_dir / "plugin_vision_test.py"
    manifest = PluginManifest.from_docstring(plugin_path)
    
    if manifest:
        missing_req, missing_opt = dep_manager.check_dependencies(manifest)
        
        print(f"\nChecking dependencies for: {manifest.plugin_name}")
        print(f"\nRequired packages ({len(manifest.required_packages)}):")
        for pkg in manifest.required_packages:
            installed = pkg not in missing_req
            status = "✓" if installed else "✗"
            print(f"  {status} {pkg}")
        
        print(f"\nOptional packages ({len(manifest.optional_packages)}):")
        for pkg in manifest.optional_packages:
            installed = pkg not in missing_opt
            status = "✓" if installed else "✗"
            print(f"  {status} {pkg}")
        
        if missing_req:
            print(f"\n⚠️  Missing required: {missing_req}")
        else:
            print(f"\n✓ All required packages installed!")
        
        if missing_opt:
            print(f"ℹ️  Missing optional: {missing_opt}")


def test_3_install_single_package():
    """Test: Install a single package"""
    print("\n" + "="*70)
    print("TEST 3: Install Single Package (pillow)")
    print("="*70)
    
    dep_manager = DependencyManager()
    
    # Check if pillow is installed
    if dep_manager.is_package_installed("pillow"):
        print("✓ Pillow is already installed")
    else:
        print("✗ Pillow not installed, attempting installation...")
        success = dep_manager.install_package("pillow")
        if success:
            print("✓ Successfully installed pillow")
        else:
            print("✗ Failed to install pillow")


def test_4_load_plugin_with_deps():
    """Test: Load plugin with dependency management"""
    print("\n" + "="*70)
    print("TEST 4: Load Plugin with Dependency Management")
    print("="*70)
    
    loader = PluginLoader(proto_dir)
    
    print("\nAttempting to load plugin_vision_test.py...")
    print("(This will check and install dependencies if needed)\n")
    
    # Load with manual confirmation (set auto_install=True to skip)
    module = loader.load_plugin("plugin_vision_test.py", auto_install=False)
    
    if module:
        print("\n✓ Plugin loaded successfully!")
        print(f"  Module: {module.__name__}")
        print(f"  Vision Available: {getattr(module, 'VISION_AVAILABLE', False)}")
    else:
        print("\n✗ Failed to load plugin")


def test_5_check_only_mode():
    """Test: Check-only mode (no installation)"""
    print("\n" + "="*70)
    print("TEST 5: Check-Only Mode")
    print("="*70)
    
    dep_manager = DependencyManager()
    plugin_path = proto_dir / "plugin_vision_test.py"
    manifest = PluginManifest.from_docstring(plugin_path)
    
    if manifest:
        missing_req, missing_opt = dep_manager.check_dependencies(manifest)
        
        print(f"\nDependency Report for: {manifest.plugin_name}")
        print("-" * 50)
        
        # Required
        print(f"\nRequired Dependencies:")
        if not missing_req:
            print("  ✓ All required packages installed")
        else:
            print("  ✗ Missing packages:")
            for pkg in missing_req:
                print(f"    - {pkg}")
        
        # Optional
        if manifest.optional_packages:
            print(f"\nOptional Dependencies:")
            if not missing_opt:
                print("  ✓ All optional packages installed")
            else:
                print("  ℹ️  Not installed:")
                for pkg in missing_opt:
                    print(f"    - {pkg}")
        
        # Overall status
        print(f"\nPlugin Status:")
        if not missing_req:
            print("  ✓ READY TO USE")
        else:
            print("  ✗ MISSING REQUIRED DEPENDENCIES")
            print(f"\n  To install, run:")
            print(f"    pip install {' '.join(missing_req)}")


def test_6_cache_system():
    """Test: Dependency cache system"""
    print("\n" + "="*70)
    print("TEST 6: Dependency Cache System")
    print("="*70)
    
    dep_manager = DependencyManager()
    
    print(f"Cache file: {dep_manager.cache_file}")
    print(f"Cached packages ({len(dep_manager.installed_cache)}):")
    
    for pkg in sorted(dep_manager.installed_cache):
        print(f"  - {pkg}")
    
    if not dep_manager.installed_cache:
        print("  (cache is empty)")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("PLUGIN DEPENDENCY MANAGER - TEST SUITE")
    print("="*70)
    
    tests = [
        ("Parse Manifest", test_1_parse_manifest),
        ("Check Dependencies", test_2_check_dependencies),
        ("Install Single Package", test_3_install_single_package),
        ("Load Plugin with Deps", test_4_load_plugin_with_deps),
        ("Check-Only Mode", test_5_check_only_mode),
        ("Cache System", test_6_cache_system),
    ]
    
    print("\nAvailable tests:")
    for i, (name, _) in enumerate(tests, 1):
        print(f"  {i}. {name}")
    
    print("\nOptions:")
    print("  all - Run all tests")
    print("  1-6 - Run specific test")
    print("  q   - Quit")
    
    while True:
        choice = input("\nSelect test: ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == 'all':
            for name, test_func in tests:
                try:
                    test_func()
                except Exception as e:
                    print(f"\n✗ Test failed with error: {e}")
                    import traceback
                    traceback.print_exc()
        elif choice.isdigit() and 1 <= int(choice) <= len(tests):
            try:
                tests[int(choice) - 1][1]()
            except Exception as e:
                print(f"\n✗ Test failed with error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
