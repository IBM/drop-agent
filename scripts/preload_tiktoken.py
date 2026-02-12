#!/usr/bin/env python3
"""
Preload tiktoken encodings for offline use.

This script downloads and caches the tiktoken encodings required by droplet,
so the application can run without internet access.
"""

import sys
import tiktoken


def preload_encodings():
    """Download and cache tiktoken encodings"""
    encodings_to_load = ["o200k_harmony", "o200k_base"]

    print("Preloading tiktoken encodings for offline use...")
    print()

    for encoding_name in encodings_to_load:
        try:
            print(f"  Loading {encoding_name}...", end=" ", flush=True)
            enc = tiktoken.get_encoding(encoding_name)
            test_tokens = enc.encode("test")
            print(f"✓ (vocab size: {enc.n_vocab})")
        except Exception as e:
            print(f"✗ Failed: {e}")
            return False

    print()
    print("✓ All tiktoken encodings loaded successfully!")
    print()
    print("The encodings are now cached and droplet can run offline.")
    return True


if __name__ == "__main__":
    success = preload_encodings()
    sys.exit(0 if success else 1)
