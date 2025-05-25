#!/usr/bin/env python3
"""
Script to check mypy type checking for the torch_relativistic package.
"""
import subprocess
import sys
import os


def run_ruff():
    """Run mypy on the src directory."""
    try:

        # Run mypy
        result = subprocess.run(
            ["uv", "run", "mypy", "src/"],
            capture_output=True,
            text=True
        )

        print("Ruff Output:")
        print("=" * 50)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        print(f"Return code: {result.returncode}")
        return result.returncode == 0

    except Exception as e:
        print(f"Error running mypy: {e}")
        return False


if __name__ == "__main__":
    success = run_ruff()
    sys.exit(0 if success else 1)
