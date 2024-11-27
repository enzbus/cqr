ENVDIR        = env
BINDIR        = $(ENVDIR)/bin

ifeq ($(OS), Windows_NT)
    BINDIR=$(ENVDIR)/Scripts
endif

.PHONY: clean test

test: env
	$(BINDIR)/python test.py

env:
	python -m venv --system-site-packages env
	$(BINDIR)/pip install -r requirements.txt

fix:
	$(BINDIR)/python -m autopep8 -i *.py
	$(BINDIR)/python -m docformatter -i *.py


clean:
	rm -rf env || true