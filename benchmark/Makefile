.PHONY: env lint fix scs bfgs hsde

env:
	python -m venv --system-site-packages env
	env/bin/pip install numpy scipy cvxpy pylint autopep8 matplotlib pandas tqdm
	# we install CQR to avoid copy-pasting too much code
	env/bin/python -m pip install -e ..
	mkdir results

lint:
	env/bin/python -m pylint benchmark
	
fix:
	env/bin/python -m autopep8 --in-place benchmark/*.py benchmark/*/*.py

brscs:
	SOLVER_CLASS=BroydenSCS env/bin/python -m benchmark.test

scs:
	SOLVER_CLASS=SimpleSCS env/bin/python -m benchmark.test

shr:
	SOLVER_CLASS=SimpleSHR env/bin/python -m benchmark.test

longscs:
	SOLVER_CLASS=LongSCS env/bin/python -m benchmark.test

eqscs:
	SOLVER_CLASS=EquilibratedSCS env/bin/python -m benchmark.test

drscs:
	SOLVER_CLASS=DouglasRachfordSCS env/bin/python -m benchmark.test

lmscs:
	SOLVER_CLASS=LevMarSCS env/bin/python -m benchmark.test

lmscsref:
	SOLVER_CLASS=LevMarRefSCS env/bin/python -m benchmark.test

lmqrscs:
	SOLVER_CLASS=LevMarQRSCS env/bin/python -m benchmark.test

lmrefqrscs:
	SOLVER_CLASS=LevMarRefQRSCS env/bin/python -m benchmark.test

bfgs:
	SOLVER_CLASS=SimpleBFGS env/bin/python -m benchmark.test

hsde:
	SOLVER_CLASS=SimpleHSDE env/bin/python -m benchmark.test

cqr:
	SOLVER_CLASS=SimpleCQR env/bin/python -m benchmark.test

uscqr:
	SOLVER_CLASS=UnscaledCQR env/bin/python -m benchmark.test