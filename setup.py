from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
def get_version():
    init_path = os.path.join(os.path.dirname(__file__), "fintics", "__init__.py")
    with open(init_path, "r", encoding="utf-8") as f:
        content = f.read()
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)
    raise RuntimeError("Unable to find version string.")

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.startswith("#") and not line.startswith("-r")
        ]

setup(
    name="fintics",
    version=get_version(),
    author="ryokobachan",
    author_email="fintics.org@gmail.com",
    description="A comprehensive framework for financial data analysis and backtesting trading strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ryokobachan/fintics",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords="trading backtesting finance strategy optimization technical-analysis",
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
    },
    entry_points={
        "console_scripts": [
            "fintics=fintics.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/ryokobachan/fintics/issues",
        "Source": "https://github.com/ryokobachan/fintics",
        "Documentation": "https://github.com/ryokobachan/fintics/blob/main/README.md",
    },
    include_package_data=True,
    zip_safe=False,
)

