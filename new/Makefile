ENVDIR        = env
PROJECT       = cqr
BINDIR        = $(ENVDIR)/bin
VENV_OPTS     =
PIP_OPTS      = --editable

ifeq ($(OS), Windows_NT)
    BINDIR = $(ENVDIR)/Scripts
    PIP_OPTS =
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		VENV_OPTS += --system-site-packages
	endif
endif

.PHONY: clean test dist

test: env
	$(BINDIR)/python -m cqr.test

env:
	python -m venv $(VENV_OPTS) $(ENVDIR)
	$(BINDIR)/pip install $(PIP_OPTS) .[test]

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