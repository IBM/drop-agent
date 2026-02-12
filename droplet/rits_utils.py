"""RITS-specific utilities for API key resolution and model listing"""

import os
import subprocess
import sys

from collections import Counter
from urllib.parse import urlparse


def resolve_api_key(provided_key, base_url):
    """
    Resolve API key for RITS endpoints from multiple sources in priority order:
    1. Provided key argument
    2. RITS_API_KEY environment variable
    3. .env file (source it and read RITS_API_KEY)

    Only applies this logic if base_url contains 'rits' (case insensitive).
    Returns the resolved API key or raises SystemExit with helpful error message.
    """
    is_rits = base_url and 'rits' in base_url.lower()

    if provided_key is not None:
        return provided_key

    if not is_rits:
        return None

    api_key = os.environ.get('RITS_API_KEY')
    if api_key:
        print("Using API key from RITS_API_KEY environment variable")
        return api_key

    env_file = '.env'
    if os.path.exists(env_file):
        print(f"Found {env_file}, sourcing it to load RITS_API_KEY...")
        result = subprocess.run(
            f'bash -c "set -a && source {env_file} && echo $RITS_API_KEY"',
            shell=True,
            capture_output=True,
            text=True
        )
        api_key = result.stdout.strip()
        if api_key:
            os.environ['RITS_API_KEY'] = api_key
            print(f"Successfully loaded RITS_API_KEY from {env_file}")
            return api_key

    print("\n\033[91mError: No API key found for RITS endpoint\033[0m\n")
    print("RITS endpoint detected. Please provide an API key using one of these methods:\n")
    print("  1. Command-line argument:")
    print("     --rits-api-key YOUR_API_KEY\n")
    print("  2. Environment variable:")
    print("     export RITS_API_KEY=YOUR_API_KEY\n")
    print("  3. Create a .env file in the current directory:")
    print("     echo 'RITS_API_KEY=YOUR_API_KEY' > .env\n")
    sys.exit(1)


def list_rits_models_and_exit(args):
    """
    List all available RITS models and exit.

    Args:
        args: Argument namespace with rits_api_key, model, tools attributes
    """
    from droplet.backend import RITSBackend
    from droplet.rich_terminal import LOGO_FAILURE, print_logo

    if not args.rits_api_key:
        print("\n\033[91mError: --rits-api-key is required to list RITS models\033[0m\n")

    else:

        try:
            print("\n🔍 Fetching available RITS models...\n")
            backend = RITSBackend(base_url="", api_key=args.rits_api_key)
            backend._fetch_available_models()

            # Count occurrences of each model name to find duplicates
            name_counts = Counter(m["model_name"] for m in backend.raw_model_data)

            # Filter models: for non-unique names, only keep if basename matches
            filtered_models = []
            for model in backend.raw_model_data:
                model_name = model["model_name"]
                endpoint = model["endpoint"]

                # If name is unique, always include it
                if name_counts[model_name] == 1:
                    filtered_models.append(model)
                else:
                    # Name appears multiple times - check if basename matches
                    parsed_url = urlparse(endpoint)
                    basename = parsed_url.path.strip('/').split('/')[-1]

                    # Check if basename matches the model name (or last part after /)
                    model_basename = model_name.split('/')[-1]
                    if basename == model_basename:
                        filtered_models.append(model)

            # Sort by model name for consistent display
            sorted_models = sorted(filtered_models, key=lambda m: m["model_name"])

            print(f"Available RITS models ({len(sorted_models)} total):\n")
            for model in sorted_models:
                model_name = model["model_name"]
                endpoint = model["endpoint"]
                print(f"  • {model_name}")
                print(f"    {endpoint}")
            print()

        except Exception as e:
            error_msg = str(e)
            # Don't show HTML error pages, just show a clean error message
            if "<html" in error_msg.lower():
                error_msg = "Service unavailable (received HTML error page)"
            print_logo(
                model=f"\033[91m{args.model}\033[0m",
                backend="\033[91mRITSBackend (https://rits.fmaas.res.ibm.com)\033[0m",
                tools=args.tools,
                logo=LOGO_FAILURE
            )
            print(f"\n\033[91mError: {error_msg}\033[0m\n")
