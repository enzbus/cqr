.PHONY: env lint fix

env:
	python -m venv --system-site-packages env
	env/bin/pip install numpy scipy cvxpy pylint autopep8

lint:
	env/bin/pylint benchmark
	
fix:
	env/bin/autopep8 --in-place benchmark/*.py 
