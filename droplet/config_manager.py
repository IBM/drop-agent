"""Configuration management for droplet"""

import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".droplet"
CONFIG_FILE = CONFIG_DIR / "config.json"


def ensure_config_dir():
    """Ensure the config directory exists"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_configs():
    """Load all configurations from file"""
    if not CONFIG_FILE.exists():
        return {}

    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)


def save_config(config_name, args):
    """
    Save configuration to file

    Args:
        config_name: Name of the configuration (use None for default)
        args: Namespace object from argparse
    """
    ensure_config_dir()

    # Load existing configs
    configs = load_configs()

    # Use original args (before config merge) if available
    if hasattr(args, '_original_args') and hasattr(args, '_explicitly_provided'):
        original_args = args._original_args
        explicitly_provided = args._explicitly_provided

        # Only save arguments that were explicitly provided
        config_dict = {}
        for key, value in original_args.items():
            # Skip config management arguments and internal attributes
            if key in ['save_config', 'load_config', 'list_configs', 'rits_list_models', '_original_args', '_defaults', '_explicitly_provided']:
                continue

            # Only save if explicitly provided on command line
            if key in explicitly_provided:
                config_dict[key] = value
    else:
        # Fallback: save all args (old behavior)
        config_dict = {}
        for key, value in vars(args).items():
            # Skip config management arguments
            if key in ['save_config', 'load_config', 'list_configs', 'rits_list_models']:
                continue
            config_dict[key] = value

    # Save under the specified name
    configs[config_name if config_name else "None"] = config_dict

    # Write back to file with indentation
    with open(CONFIG_FILE, 'w') as f:
        json.dump(configs, f, indent=2)

    print(f"\n✓ Configuration saved as '{config_name if config_name else 'None'}' to {CONFIG_FILE}\n")


def load_config(config_name):
    """
    Load a specific configuration

    Args:
        config_name: Name of the configuration to load

    Returns:
        Dictionary of configuration values or None if not found
    """
    configs = load_configs()
    return configs.get(config_name)


def list_configs():
    """List all available configurations"""
    configs = load_configs()

    print(f"\nConfig path: {CONFIG_FILE}\n")

    if not configs:
        print("No saved configurations found.\n")
        return

    print("Saved configurations:\n")
    for name, config in sorted(configs.items()):
        print(f"  • {name}")
        # Show key settings
        if 'model' in config:
            print(f"    model: {config['model']}")
        if 'backend_type' in config:
            print(f"    backend: {config['backend_type']}")
        if 'backend_url' in config and config['backend_type'] != 'rits-vllm':
            print(f"    url: {config['backend_url']}")
        if 'tools' in config:
            print(f"    tools: {', '.join(config['tools'])}")
        print()

    print("To select a config: droplet -c <name>\n")
