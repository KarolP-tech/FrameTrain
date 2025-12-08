#!/bin/bash

# Setup Virtual Environment for FrameTrain Development
# This ensures VS Code and the app use the same Python interpreter

set -e

echo "🐍 Setting up Python Virtual Environment for FrameTrain..."
echo ""

# Navigate to project root
cd "$(dirname "$0")"

# Check if venv exists
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists at ./venv"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing venv. Activating it..."
        source venv/bin/activate
        echo "✅ Virtual environment activated"
        echo "Python: $(which python3)"
        echo "Pip: $(which pip3)"
        exit 0
    else
        echo "Removing old venv..."
        rm -rf venv
    fi
fi

# Create new virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate it
source venv/bin/activate

echo "✅ Virtual environment created and activated"
echo "Python: $(which python3)"

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install base dependencies (common for all plugins)
echo ""
echo "📦 Installing base dependencies..."
pip install torch torchvision transformers datasets

# Read installed plugins from state file
STATE_FILE="$HOME/.frametrain/plugin_state.json"

if [ -f "$STATE_FILE" ]; then
    echo ""
    echo "📋 Found installed plugins, installing their dependencies..."
    
    # Parse JSON and install dependencies
    # This is a simple bash approach - could be improved with jq
    
    # For now, install common dependencies
    echo "Installing common ML packages..."
    pip install \
        torch \
        torchvision \
        torchaudio \
        transformers \
        datasets \
        timm \
        pillow \
        numpy \
        pandas \
        scikit-learn \
        matplotlib
    
    echo ""
    echo "✅ Common dependencies installed"
else
    echo ""
    echo "ℹ️  No plugin state found. Install base packages only."
    echo "   Run the app once to install plugin-specific dependencies."
fi

# Create .vscode/settings.json if it doesn't exist
echo ""
echo "⚙️  Configuring VS Code..."

mkdir -p .vscode

cat > .vscode/settings.json << EOF
{
  "python.defaultInterpreterPath": "\${workspaceFolder}/venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.analysis.extraPaths": [
    "\${workspaceFolder}/src-tauri/python",
    "\${workspaceFolder}/src-tauri/python/train_engine",
    "\${workspaceFolder}/src-tauri/python/test_engine"
  ],
  "python.autoComplete.extraPaths": [
    "\${workspaceFolder}/src-tauri/python",
    "\${workspaceFolder}/src-tauri/python/train_engine",
    "\${workspaceFolder}/src-tauri/python/test_engine"
  ]
}
EOF

echo "✅ VS Code configured to use virtual environment"

# Update plugin_commands.rs to use venv Python
echo ""
echo "📝 Note: To use this venv in the app, update plugin_commands.rs:"
echo "   Change get_python_executable() to return: $(pwd)/venv/bin/python3"

echo ""
echo "============================================"
echo "✅ Setup Complete!"
echo "============================================"
echo ""
echo "📌 Next steps:"
echo "   1. Restart VS Code to apply settings"
echo "   2. All yellow import warnings should disappear"
echo "   3. To activate venv manually: source venv/bin/activate"
echo ""
echo "🔧 To use this Python in the app:"
echo "   Update src-tauri/src/plugin_commands.rs:"
echo "   fn get_python_executable() -> \"$(pwd)/venv/bin/python3\""
echo ""
