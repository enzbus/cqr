.PHONY: env lint fix

env:
	python -m venv --system-site-packages env
	env/bin/pip install numpy scipy cvxpy pylint autopep8

lint:
	env/bin/python -m pylint benchmark
	
fix:
	env/bin/python -m autopep8 --in-place benchmark/*.py benchmark/*/*.py

test:
	env/bin/python -m benchmark.test