#!/bin/bash

# Rust Installation Script für FrameTrain

echo "🦀 Rust Installation für FrameTrain"
echo "===================================="
echo ""

# Prüfe ob Rust bereits installiert ist
if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version)
    echo "✓ Rust ist bereits installiert: $RUST_VERSION"
    echo ""
    read -p "Möchtest du Rust aktualisieren? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rustup update
        echo "✓ Rust aktualisiert"
    fi
else
    echo "→ Installiere Rust..."
    echo ""
    
    # Installiere Rust via rustup
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    
    # Lade Rust in aktuelle Shell
    source "$HOME/.cargo/env"
    
    echo ""
    echo "✓ Rust erfolgreich installiert!"
    echo ""
fi

# Verifiziere Installation
if command -v rustc &> /dev/null; then
    echo "✓ Rust Version: $(rustc --version)"
    echo "✓ Cargo Version: $(cargo --version)"
    echo ""
    echo "🎉 Rust ist bereit!"
    echo ""
    echo "Du kannst jetzt das Setup fortsetzen:"
    echo "  ./setup.sh"
else
    echo "❌ Rust Installation fehlgeschlagen"
    echo "Bitte installiere Rust manuell: https://rustup.rs/"
    exit 1
fi
