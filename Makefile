# This Makefile is used to automate some development tasks.
# Ideally this logic would be in pyproject.toml but it appears
# easier to do it this way for now.

PYTHON			= python
PROJECT			= project_euromir
TESTSDIR		= $(PROJECT)/tests
DOCSDIR			= docs
DOCBUILDDIR		= $(DOCSDIR)/_build
BUILDDIR		= build
ENVDIR			= env
BINDIR			= $(ENVDIR)/bin
VENV_OPTS		=
CMAKE_OPTS		=

# Python venv on Windows has different location
ifeq ($(OS), Windows_NT)
	BINDIR=$(ENVDIR)/Scripts
	CMAKE_OPTS = -G "MinGW Makefiles"
# on Linux we use system-installed packages
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		VENV_OPTS += --system-site-packages
	endif
	export PATH := BINDIR:$(PATH)
endif

.PHONY: env clean update test lint docs fix release build pybuild

env: ## create environment
	$(PYTHON) -m venv $(VENV_OPTS) $(ENVDIR)
	$(BINDIR)/python -m pip install -v -e .[dev,tests,docs]

# build: ## build locally (instead of editable install)
# 	cmake -B$(BUILDDIR)  $(CMAKE_OPTS)
# 	cmake --build $(BUILDDIR)
# 	cmake --install $(BUILDDIR)

clean:  ## clean environment
	-rm -rf $(DOCBUILDDIR)
	-rm -rf $(BUILDDIR)
	-rm -rf $(ENVDIR)
	-rm -rf $(PROJECT).egg-info
	-rm $(PROJECT)/*.dylib
	-rm $(PROJECT)/*.so
	-rm $(PROJECT)/*.dll


update: clean env  ## clean and recreate environment

pybuild: ## create Python packages
	python -m build
	python setup.py build # only used to get platform tag
	$(eval TAG=`python -c "from wheel.bdist_wheel import get_platform; print(get_platform('build/lib'))"`)
	python -m wheel tags dist/*.whl --platform="${TAG}" --remove

test: #build ## run unit tests
	$(BINDIR)/python -m coverage run -m $(PROJECT).tests
	$(BINDIR)/python -m coverage report
	$(BINDIR)/python -m coverage xml
	$(BINDIR)/diff-cover coverage.xml --config-file pyproject.toml

coverage:  ## open html cov report
	$(BINDIR)/python -m coverage html --fail-under=0 # overwrite pyproject.toml default
	open htmlcov/index.html

lint:  ## run Pylint
	pylint $(PROJECT)

docs:  ## build Sphinx docs
	sphinx-build -E docs $(DOCBUILDDIR)
	open $(DOCBUILDDIR)/index.html

fix:  ## auto-fix Python code
	# selected among many code auto-fixers, tweaked in pyproject.toml
	autopep8 -i -r $(PROJECT)
	python -m isort $(PROJECT)
	# this is the best found for the purpose
	docformatter -r --in-place $(PROJECT)

# release: update lint test  ## update version, publish to PyPI
# 	python -m build
#	twine check dist/*
#	twine upload --skip-existing dist/*

# Thanks to Francoise at marmelab.com for this
.DEFAULT_GOAL := help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

print-%:
	@echo '$*=$($*)'
