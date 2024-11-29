ENVDIR        = env
BINDIR        = $(ENVDIR)/bin

ifeq ($(OS), Windows_NT)
    BINDIR=$(ENVDIR)/Scripts
endif

.PHONY: clean test

test: env
	$(BINDIR)/python -m cqr.test

env:
	python -m venv --system-site-packages env
	$(BINDIR)/pip install --editable .[test]

fix:
	$(BINDIR)/python -m autopep8 -i cqr/*.py
	$(BINDIR)/python -m docformatter -i cqr/*.py


clean:
	rm -rf env || true