from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fintics",
    version="1.0.0",
    author="ryokobachan",
    author_email="",
    description="A comprehensive framework for financial data analysis and backtesting trading strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ryokobachan/fintics",
    packages=find_packages(),
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
    ],
    python_requires=">=3.9",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#") and "dependencies" not in line
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "fintics=fintics.cli:main",  # 将来的なCLI対応
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/ryokobachan/fintics/issues",
        "Source": "https://github.com/ryokobachan/fintics",
    },
)
