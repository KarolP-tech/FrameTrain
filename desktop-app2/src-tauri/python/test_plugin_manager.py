#!/usr/bin/env python3
"""
Quick test for plugin_manager.py
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from plugin_manager import PluginManager
    
    print("✅ Plugin manager imported successfully")
    
    # Create manager
    manager = PluginManager()
    
    # Test getting plugins for first launch
    plugins = manager.get_plugins_for_first_launch()
    
    print(f"✅ Found {len(plugins)} plugins")
    
    # Output as JSON (like the CLI does)
    json_output = json.dumps(plugins, indent=2)
    print("\n📋 JSON Output (first 500 chars):")
    print(json_output[:500] + "...")
    
    # Verify structure
    if plugins:
        first_plugin = plugins[0]
        required_fields = ['id', 'name', 'description', 'icon', 'estimated_size_mb']
        missing_fields = [f for f in required_fields if f not in first_plugin]
        
        if missing_fields:
            print(f"❌ Missing fields in plugin: {missing_fields}")
            sys.exit(1)
        else:
            print(f"✅ Plugin structure is valid")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
