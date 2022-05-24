
init:
	pip install --upgrade pip
	pip install poetry
	npm install -g git-changelog

install:
	poetry install

test:
	python -m pytest --cov-report html:./coverage.html --cov=pyskoptimize tests/
