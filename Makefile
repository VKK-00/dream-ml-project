.PHONY: fmt lint nbclean checks

fmt:
	ruff check . --fix
	black .

lint:
	ruff check .

nbclean:
	nbstripout --install
	nbstripout --strip --force $(shell git ls-files '*.ipynb')

checks:
	pre-commit run --all-files --show-diff-on-failure
