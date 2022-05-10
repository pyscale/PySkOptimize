
init:
	pip install --upgrade pip
	pip install poetry

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest
