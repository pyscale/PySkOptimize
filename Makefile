
init:
	pip install --upgrade pip
	pip install poetry

install:
	poetry install

test:
	python -m pytest
