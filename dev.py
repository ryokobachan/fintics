#!/usr/bin/env python3
"""Development utility script for common tasks."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def run_tests():
    """Run all tests with coverage."""
    cmd = [
        "pytest", 
        "tests/", 
        "-v", 
        "--cov=fintics", 
        "--cov-report=html",
        "--cov-report=term"
    ]
    subprocess.run(cmd, cwd=ROOT)


def format_code():
    """Format code using black and isort."""
    subprocess.run(["black", "fintics/", "tests/"], cwd=ROOT)
    subprocess.run(["isort", "fintics/", "tests/"], cwd=ROOT)


def lint_code():
    """Run linting checks."""
    subprocess.run(["flake8", "fintics/", "tests/"], cwd=ROOT)
    subprocess.run(["mypy", "fintics/"], cwd=ROOT)


def setup_dev():
    """Setup development environment."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"], cwd=ROOT)
    subprocess.run(["pre-commit", "install"], cwd=ROOT)


def clean():
    """Clean build artifacts."""
    import shutil
    for pattern in ["build", "dist", "*.egg-info", "__pycache__", ".pytest_cache"]:
        for path in ROOT.rglob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Development utilities for Fintics")
    parser.add_argument("command", choices=["test", "format", "lint", "setup", "clean"])
    
    args = parser.parse_args()
    
    commands = {
        "test": run_tests,
        "format": format_code,
        "lint": lint_code,
        "setup": setup_dev,
        "clean": clean
    }
    
    commands[args.command]()
