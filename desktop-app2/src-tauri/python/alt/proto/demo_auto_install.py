#!/usr/bin/env python3
"""
Quick Demo: Automatische Dependency Installation
=================================================

Dieses Skript demonstriert das automatische Dependency Management.
"""

import os
import sys
from pathlib import Path

# Add proto dir to path
proto_dir = Path(__file__).parent
sys.path.insert(0, str(proto_dir))

print("="*70)
print("PROTO TRAIN ENGINE - DEPENDENCY AUTO-INSTALL DEMO")
print("="*70)

def demo_1_check_plugin_deps():
    """Demo: Check welche Dependencies ein Plugin braucht"""
    print("\n" + "="*70)
    print("DEMO 1: Plugin Dependencies prüfen")
    print("="*70)
    
    from plugin_dependency_manager import PluginManifest, DependencyManager
    
    plugin_file = proto_dir / "plugin_vision.py"
    
    print(f"\nPrüfe Plugin: {plugin_file.name}")
    
    # Manifest lesen
    manifest = PluginManifest.from_docstring(plugin_file)
    
    if manifest:
        print(f"\n✓ Manifest gefunden:")
        print(f"  Name: {manifest.plugin_name}")
        print(f"  Description: {manifest.description}")
        print(f"  Required: {manifest.required_packages}")
        print(f"  Optional: {manifest.optional_packages}")
        
        # Dependencies checken
        dep_manager = DependencyManager()
        missing_req, missing_opt = dep_manager.check_dependencies(manifest)
        
        print(f"\n📊 Dependency Status:")
        print(f"  Required packages:")
        for pkg in manifest.required_packages:
            installed = pkg not in missing_req
            status = "✓" if installed else "✗"
            print(f"    {status} {pkg}")
        
        if missing_req:
            print(f"\n⚠️  Fehlende Required: {missing_req}")
        else:
            print(f"\n✓ Alle Required packages installiert!")


def demo_2_load_plugin_with_auto_install():
    """Demo: Plugin mit Auto-Install laden"""
    print("\n" + "="*70)
    print("DEMO 2: Plugin mit Auto-Install laden")
    print("="*70)
    
    from plugin_dependency_manager import PluginLoader
    
    # Create loader
    loader = PluginLoader(str(proto_dir))
    
    # Choose a plugin to test
    plugin_name = "plugin_vision.py"
    
    print(f"\nLade Plugin: {plugin_name}")
    print(f"Auto-Install: True (Dependencies werden automatisch installiert)\n")
    
    # Load with auto-install
    module = loader.load_plugin(plugin_name, auto_install=True)
    
    if module:
        print(f"\n✅ Plugin erfolgreich geladen!")
        print(f"   Module: {module.__name__}")
        print(f"   Available: {getattr(module, 'TIMM_AVAILABLE', 'Unknown')}")
    else:
        print(f"\n❌ Plugin konnte nicht geladen werden")


def demo_3_training_engine_integration():
    """Demo: Integration in Training Engine"""
    print("\n" + "="*70)
    print("DEMO 3: Training Engine Integration")
    print("="*70)
    
    print("\nDies zeigt wie das System im echten Training funktioniert:\n")
    
    print("1. User startet Training mit Vision-Dataset")
    print("2. System erkennt Modalität: VISION")
    print("3. System prüft ob plugin_vision.py geladen ist")
    print("4. Falls nicht: Manifest lesen, Dependencies prüfen")
    print("5. Fehlende Pakete installieren (mit Auto-Install)")
    print("6. Plugin laden und registrieren")
    print("7. Training starten")
    
    print("\n💡 Um das live zu sehen:")
    print("   export FRAMETRAIN_AUTO_INSTALL=true")
    print("   python proto_train_engine.py --config your_config.json")


def demo_4_env_vars():
    """Demo: Environment Variables"""
    print("\n" + "="*70)
    print("DEMO 4: Environment Variables")
    print("="*70)
    
    print("\nVerfügbare Environment Variables:\n")
    
    # Check current settings
    auto_install = os.getenv("FRAMETRAIN_AUTO_INSTALL", "not set")
    debug = os.getenv("FRAMETRAIN_DEBUG", "not set")
    
    print(f"FRAMETRAIN_AUTO_INSTALL: {auto_install}")
    print(f"  → Steuert automatische Dependency-Installation")
    print(f"  → 'true' = Auto-Install ohne Fragen")
    print(f"  → 'false' oder nicht gesetzt = User wird gefragt")
    
    print(f"\nFRAMETRAIN_DEBUG: {debug}")
    print(f"  → Aktiviert Debug-Ausgaben")
    print(f"  → 'true' = Mehr Logging")
    
    print("\n💡 Setzen:")
    print("   export FRAMETRAIN_AUTO_INSTALL=true")
    print("   export FRAMETRAIN_DEBUG=true")


def demo_5_cache_info():
    """Demo: Cache System"""
    print("\n" + "="*70)
    print("DEMO 5: Dependency Cache")
    print("="*70)
    
    from plugin_dependency_manager import DependencyManager
    
    dep_manager = DependencyManager()
    
    print(f"\nCache Location: {dep_manager.cache_file}")
    print(f"Cached Packages: {len(dep_manager.installed_cache)}")
    
    if dep_manager.installed_cache:
        print("\nCached Dependencies:")
        for pkg in sorted(dep_manager.installed_cache):
            print(f"  ✓ {pkg}")
    else:
        print("\n(Cache ist leer - wird beim ersten Check gefüllt)")
    
    print("\n💡 Cache löschen:")
    print(f"   rm {dep_manager.cache_file}")


def main():
    """Run all demos"""
    
    demos = [
        ("Plugin Dependencies prüfen", demo_1_check_plugin_deps),
        ("Plugin mit Auto-Install laden", demo_2_load_plugin_with_auto_install),
        ("Training Engine Integration", demo_3_training_engine_integration),
        ("Environment Variables", demo_4_env_vars),
        ("Dependency Cache", demo_5_cache_info),
    ]
    
    print("\nVerfügbare Demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    
    print("\nOptionen:")
    print("  all   - Alle Demos ausführen")
    print("  1-5   - Einzelne Demo ausführen")
    print("  q     - Beenden")
    
    while True:
        choice = input("\nDemo auswählen: ").strip().lower()
        
        if choice == 'q':
            print("\n👋 Tschüss!")
            break
        
        elif choice == 'all':
            for name, demo_func in demos:
                try:
                    demo_func()
                except Exception as e:
                    print(f"\n❌ Demo fehlgeschlagen: {e}")
                    import traceback
                    traceback.print_exc()
        
        elif choice.isdigit() and 1 <= int(choice) <= len(demos):
            try:
                demos[int(choice) - 1][1]()
            except Exception as e:
                print(f"\n❌ Demo fehlgeschlagen: {e}")
                import traceback
                traceback.print_exc()
        
        else:
            print("❌ Ungültige Auswahl")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Abgebrochen!")
        sys.exit(0)
