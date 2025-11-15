"""
Main CLI entry point for FrameTrain
"""

import click
import sys
from colorama import init, Fore, Style

from .commands import install, start, update, verify_key, config
from .utils import print_banner, print_error, print_success

# Initialize colorama for cross-platform colored output
init(autoreset=True)


@click.group()
@click.version_option(version="1.0.0", prog_name="frametrain")
def main():
    """
    FrameTrain CLI - Professional Platform for Local ML Training
    
    Use 'frametrain COMMAND --help' for more information on a command.
    """
    pass


@main.command()
@click.option('--key', prompt='Enter your API key', help='Your FrameTrain API key')
@click.option('--path', default=None, help='Installation directory (default: ~/frametrain)')
def install(key: str, path: str):
    """Install the FrameTrain desktop application"""
    print_banner()
    install.install_app(key, path)


@main.command()
@click.option('--verify/--no-verify', default=True, help='Verify key before starting')
def start(verify: bool):
    """Start the FrameTrain desktop application"""
    start.start_app(verify)


@main.command()
@click.option('--force', is_flag=True, help='Force update even if already up-to-date')
def update(force: bool):
    """Update the FrameTrain desktop application"""
    print_banner()
    update.update_app(force)


@main.command('verify-key')
@click.option('--key', prompt='Enter your API key', help='API key to verify')
def verify_key_cmd(key: str):
    """Verify an API key"""
    verify_key.verify_api_key(key)


@main.command()
@click.argument('action', type=click.Choice(['show', 'set-key', 'set-url']))
@click.option('--key', help='API key (for set-key action)')
@click.option('--url', help='API URL (for set-url action)')
def config_cmd(action: str, key: str, url: str):
    """Manage FrameTrain configuration"""
    config.manage_config(action, key, url)


@main.command()
def info():
    """Show FrameTrain installation and configuration info"""
    print_banner()
    config.show_info()


@main.command()
@click.confirmation_option(prompt='Are you sure you want to uninstall FrameTrain?')
def uninstall():
    """Uninstall FrameTrain desktop application"""
    install.uninstall_app()


if __name__ == '__main__':
    main()
