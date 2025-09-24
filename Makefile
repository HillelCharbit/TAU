.PHONY: lint test coverage build clean

lint:
	ruff check src tests

test:
	pytest --maxfail=1 --disable-warnings

coverage:
	pytest --cov=tau_community_detection --cov-report=term-missing --cov-report=xml

build:
	python -m build

clean:
	rm -rf build dist *.egg-info .pytest_cache
