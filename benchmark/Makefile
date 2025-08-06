.PHONY: env lint fix scs bfgs hsde

env:
	python -m venv --system-site-packages env
	env/bin/pip install numpy scipy cvxpy pylint autopep8 matplotlib
	# we install CQR to avoid copy-pasting too much code
	env/bin/python -m pip install -e ..
	mkdir results

lint:
	env/bin/python -m pylint benchmark
	
fix:
	env/bin/python -m autopep8 --in-place benchmark/*.py benchmark/*/*.py

scs:
	SOLVER_CLASS=SimpleSCS env/bin/python -m benchmark.test

eqscs:
	SOLVER_CLASS=EquilibratedSCS env/bin/python -m benchmark.test

bfgs:
	SOLVER_CLASS=SimpleBFGS env/bin/python -m benchmark.test

hsde:
	SOLVER_CLASS=SimpleHSDE env/bin/python -m benchmark.test

cqr:
	SOLVER_CLASS=SimpleCQR env/bin/python -m benchmark.test