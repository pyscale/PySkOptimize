

install:

	python -m pip install -r requirements.txt

docker.build:

	docker compose build --force-rm

docker.test:

	docker compose up

heroku.deploy:

	git push heroku main