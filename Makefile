.PHONY: lint test coverage build clean verify-dist

lint:
	ruff check --select E9,F63,F7,F82 src tests

test:
	pytest --maxfail=1 --disable-warnings

coverage:
	pytest --cov=tau_community_detection --cov-report=term-missing --cov-report=xml

build:
	python -m build

verify-dist: clean build
	python3 scripts/verify_dist.py

clean:
	rm -rf build dist *.egg-info .pytest_cache
