#!/usr/bin/env python3
"""
Quick Start - Plugin Dependency Manager Demo
=============================================

Einfaches Beispiel zur Demonstration des Systems.
"""

import sys
from pathlib import Path

# Add proto directory to path
proto_dir = Path(__file__).parent
sys.path.insert(0, str(proto_dir))

print("=" * 70)
print("PLUGIN DEPENDENCY MANAGER - QUICK START DEMO")
print("=" * 70)

# ============================================================================
# DEMO 1: Parse Plugin Manifest
# ============================================================================

print("\n📋 DEMO 1: Plugin Manifest Parser")
print("-" * 70)

from plugin_dependency_manager import PluginManifest

plugin_path = proto_dir / "plugin_vision_test.py"
manifest = PluginManifest.from_docstring(plugin_path)

if manifest:
    print(f"✓ Plugin: {manifest.plugin_name}")
    print(f"  Description: {manifest.description}")
    print(f"  Modality: {manifest.modality}")
    print(f"  Required: {', '.join(manifest.required_packages)}")
    print(f"  Optional: {', '.join(manifest.optional_packages)}")
else:
    print("✗ Failed to parse manifest")

# ============================================================================
# DEMO 2: Check Dependencies
# ============================================================================

print("\n🔍 DEMO 2: Check What's Installed")
print("-" * 70)

from plugin_dependency_manager import DependencyManager

dep_manager = DependencyManager()

if manifest:
    missing_required, missing_optional = dep_manager.check_dependencies(manifest)
    
    print(f"\nRequired packages:")
    for pkg in manifest.required_packages:
        installed = pkg not in missing_required
        status = "✓ Installed" if installed else "✗ Missing"
        print(f"  {pkg:20s} - {status}")
    
    print(f"\nOptional packages:")
    for pkg in manifest.optional_packages:
        installed = pkg not in missing_optional
        status = "✓ Installed" if installed else "✗ Missing"
        print(f"  {pkg:20s} - {status}")
    
    if not missing_required:
        print("\n✓ All required dependencies are installed!")
    else:
        print(f"\n⚠️  Missing required: {', '.join(missing_required)}")

# ============================================================================
# DEMO 3: Load Plugin
# ============================================================================

print("\n🚀 DEMO 3: Load Plugin with Dependency Management")
print("-" * 70)

from plugin_dependency_manager import PluginLoader

loader = PluginLoader(proto_dir, dep_manager)

print("\nAttempting to load plugin_vision_test.py...")

# For demo purposes, we'll check without installing
if manifest:
    missing_req, missing_opt = dep_manager.check_dependencies(manifest)
    
    if missing_req:
        print(f"\n⚠️  Cannot load plugin - missing required dependencies:")
        for pkg in missing_req:
            print(f"  - {pkg}")
        
        print(f"\nTo install dependencies, run:")
        print(f"  pip install {' '.join(missing_req)}")
        
        print(f"\nOr use the test script to install interactively:")
        print(f"  python test_dependency_manager.py")
    else:
        print("\n✓ All dependencies available!")
        print("Plugin can be loaded safely.")
        
        # Uncomment to actually load:
        # module = loader.load_plugin("plugin_vision_test.py", auto_install=False)
        # if module:
        #     print(f"✓ Successfully loaded: {module.__name__}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
✓ Demonstrated:
  1. Parsing plugin manifests from docstrings
  2. Checking installed vs missing dependencies
  3. Loading plugins with dependency management

📚 Next Steps:
  1. Run full test suite: python test_dependency_manager.py
  2. Read documentation: PLUGIN_DEPENDENCY_README.md
  3. Create your own plugin with a manifest

🔧 Quick Commands:
  # Check dependencies only
  python plugin_dependency_manager.py --plugin plugin_vision_test.py --check-only
  
  # Load with auto-install
  python plugin_dependency_manager.py --plugin plugin_vision_test.py --auto-install
  
  # Run interactive tests
  python test_dependency_manager.py
""")

print("=" * 70)
