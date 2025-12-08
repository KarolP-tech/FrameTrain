"""
Enhanced Plugin Loading for TrainingEngine
==========================================
Add this code to the TrainingEngine class in train_engine.py
to enable automatic plugin management with user consent.

Replace the _ensure_plugin_loaded method with this enhanced version.
"""

def _ensure_plugin_loaded(self, modality: Modality):
    """
    Ensure the appropriate plugin is loaded for the detected modality.
    Enhanced with PluginManager integration for automatic installation.
    """
    # Map modalities to plugin IDs
    plugin_id_map = {
        Modality.TEXT: "text",  # Built-in
        Modality.VISION: "vision",
        Modality.AUDIO: "audio",
        Modality.GRAPH: "graph",
        Modality.TABULAR: "tabular",
        Modality.TIME_SERIES: "timeseries",
        Modality.MULTIMODAL: "multimodal",
        Modality.REINFORCEMENT: "rl",
    }
    
    plugin_id = plugin_id_map.get(modality)
    
    if plugin_id is None or plugin_id == "text":
        # Built-in support, no plugin needed
        MessageProtocol.debug(f"Modality {modality.value} uses built-in support")
        return
    
    # Check if plugin is already loaded in REGISTRY
    if REGISTRY.get_data_loader(modality) is not None:
        MessageProtocol.debug(f"Plugin for {modality.value} already loaded")
        return
    
    # Plugin not loaded - use PluginManager to ensure it's available
    MessageProtocol.status(
        "plugin_check",
        f"Checking plugin availability for {modality.value}..."
    )
    
    # Import PluginManager
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from plugin_manager import PluginManager
        
        plugin_manager = PluginManager()
        
        # Check if plugin is installed
        if not plugin_manager.state.is_plugin_installed(plugin_id):
            MessageProtocol.status(
                "plugin_needed",
                f"Plugin '{plugin_id}' is required but not installed"
            )
            
            # Try to ensure plugin is available (will prompt user if needed)
            success = plugin_manager.ensure_plugin_available(plugin_id, modality.value)
            
            if not success:
                raise RuntimeError(
                    f"Plugin '{plugin_id}' is required for {modality.value} but could not be installed.\n"
                    f"Please install it manually via Settings or run with --plugins flag."
                )
        
        # Plugin is installed, now load it
        MessageProtocol.status(
            "loading_plugin",
            f"Loading plugin for {modality.value}..."
        )
        
        # Map plugin ID to plugin file
        plugin_file_map = {
            "vision": "plugin_vision.py",
            "audio": "plugin_audio.py",
            "detection": "plugin_detection.py",
            "graph": "plugin_graph.py",
            "tabular": "plugin_tabular.py",
            "timeseries": "plugin_timeseries.py",
            "multimodal": "plugin_multimodal.py",
            "rl": "plugin_rl.py",
            "segmentation": "plugin_segmentation.py"
        }
        
        plugin_file = plugin_file_map.get(plugin_id)
        
        if not plugin_file:
            raise RuntimeError(f"Unknown plugin ID: {plugin_id}")
        
        # Create plugin loader if not exists
        if self.plugin_loader is None:
            plugin_dir = Path(__file__).parent / "plugins"
            self.plugin_loader = PluginLoader(str(plugin_dir))
        
        # Load the plugin (dependencies should already be installed by PluginManager)
        auto_install = os.getenv("FRAMETRAIN_AUTO_INSTALL", "false").lower() == "true"
        module = self.plugin_loader.load_plugin(plugin_file, auto_install=auto_install)
        
        if module is None:
            raise RuntimeError(
                f"Failed to load plugin: {plugin_file}\n"
                f"This plugin is needed for {modality.value} tasks."
            )
        
        # Verify plugin registered correctly
        if REGISTRY.get_data_loader(modality) is None:
            raise RuntimeError(
                f"Plugin {plugin_file} loaded but did not register for {modality.value}\n"
                f"This is a bug in the plugin."
            )
        
        MessageProtocol.status(
            "plugin_ready",
            f"Plugin for {modality.value} is ready"
        )
        
    except ImportError as e:
        # Fallback to old behavior if PluginManager not available
        MessageProtocol.warning(f"PluginManager not available: {e}")
        MessageProtocol.warning("Falling back to simple plugin loading")
        
        # Simple loading without dependency check
        plugin_file_map = {
            Modality.VISION: "plugin_vision.py",
            Modality.AUDIO: "plugin_audio.py",
            Modality.GRAPH: "plugin_graph.py",
            Modality.TABULAR: "plugin_tabular.py",
            Modality.TIME_SERIES: "plugin_timeseries.py",
            Modality.MULTIMODAL: "plugin_multimodal.py",
            Modality.REINFORCEMENT: "plugin_rl.py",
        }
        
        plugin_file = plugin_file_map.get(modality)
        
        if self.plugin_loader is None:
            plugin_dir = Path(__file__).parent / "plugins"
            self.plugin_loader = PluginLoader(str(plugin_dir))
        
        auto_install = os.getenv("FRAMETRAIN_AUTO_INSTALL", "false").lower() == "true"
        module = self.plugin_loader.load_plugin(plugin_file, auto_install=auto_install)
        
        if module is None:
            raise RuntimeError(
                f"Failed to load required plugin: {plugin_file}\n"
                f"This plugin is needed for {modality.value} tasks."
            )
