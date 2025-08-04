.PHONY: env lint fix scs bfgs hsde

env:
	python -m venv --system-site-packages env
	env/bin/pip install numpy scipy cvxpy pylint autopep8 matplotlib
	mkdir results

lint:
	env/bin/python -m pylint benchmark
	
fix:
	env/bin/python -m autopep8 --in-place benchmark/*.py benchmark/*/*.py

scs:
	SOLVER_CLASS=SimpleSCS env/bin/python -m benchmark.test

bfgs:
	SOLVER_CLASS=SimpleBFGS env/bin/python -m benchmark.test

hsde:
	SOLVER_CLASS=SimpleHSDE env/bin/python -m benchmark.test