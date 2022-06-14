
init:
	pip install --upgrade pip
	pip install poetry
	npm install -g git-changelog

install:
	poetry install --no-root

test:
	python -m pytest -v --cov-report=xml --cov=pyskoptimize tests/
