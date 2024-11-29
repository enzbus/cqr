ENVDIR        = env
PROJECT       = cqr
BINDIR        = $(ENVDIR)/bin

ifeq ($(OS), Windows_NT)
    BINDIR=$(ENVDIR)/Scripts
endif

.PHONY: clean test dist

test: env
	$(BINDIR)/python -m cqr.test

env:
	python -m venv --system-site-packages env
	$(BINDIR)/pip install --upgrade pip
	$(BINDIR)/pip install --editable .[test]

fix:
	$(BINDIR)/python -m autopep8 -i cqr/*.py
	$(BINDIR)/python -m docformatter -i cqr/*.py

lint:  ## run linter
	$(BINDIR)/python -m pylint $(PROJECT)
	$(BINDIR)/diff-quality --violations=pylint --config-file pyproject.toml

dist: ## update version, publish to pypi
	$(BINDIR)/python -m rstcheck README.rst
	$(BINDIR)/python -m build
	$(BINDIR)/python -m twine check dist/*

publish: ## publish to pypi
	$(BINDIR)/python -m twine upload --skip-existing dist/*


clean:
	rm -rf env || true