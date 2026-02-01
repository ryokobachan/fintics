.PHONY: help install install-dev test lint format clean setup build
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package
	pip install -r requirements.txt
	pip install -e .

install-dev: ## Install package in development mode
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

test: ## Run tests
	python dev.py test

lint: ## Run linting checks
	python dev.py lint

format: ## Format code
	python dev.py format

clean: ## Clean build artifacts
	python dev.py clean

setup: ## Setup development environment
	python dev.py setup

build: ## Build distribution packages
	python setup.py sdist bdist_wheel

check-deps: ## Check for outdated dependencies
	pip list --outdated

update-deps: ## Update all dependencies
	pip install --upgrade pip
	pip install --upgrade -r requirements.txt

publish-test: ## Publish to test PyPI
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

publish: ## Publish to PyPI
	twine upload dist/*

docs: ## Generate documentation (placeholder)
	@echo "Documentation generation not implemented yet"

benchmark: ## Run performance benchmarks (placeholder)
	@echo "Benchmarks not implemented yet"
