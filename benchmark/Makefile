.PHONY: env lint fix scs bfgs hsde

export OPENBLAS_NUM_THREADS=1

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

realscs:
	SOLVER_CLASS=RealSCS NUM_INSTANCES=1000 env/bin/python -m benchmark.test

realosqp:
	SOLVER_CLASS=RealOSQP NUM_INSTANCES=1000 SIZE_CHOICE=NORMAL env/bin/python -m benchmark.test

newcqr:
	SOLVER_CLASS=NewCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

nonsymmsoc:
	SOLVER_CLASS=EquilibratedNewCQRNonSymmSOC NUM_INSTANCES=1000 env/bin/python -m benchmark.test

ntest0brcqr: # w/out eq
	SOLVER_CLASS=SparseNTestBroydenCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

ntestbrcqr: # best prototype so far
	SOLVER_CLASS=SparseNTestEquilibratedBroydenCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

ntest1brcqr: # test with more ruiz 
	SOLVER_CLASS=SparseNTestEquilibrated1BroydenCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

ntest2brcqr: # test with less accel cap
	SOLVER_CLASS=SparseNTestEquilibrated2BroydenCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

ntest3brcqr: # test with more accel cap
	SOLVER_CLASS=SparseNTestEquilibrated3BroydenCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

ntest4brcqr: # test with more memory, less cap
	SOLVER_CLASS=SparseNTestEquilibrated4BroydenCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

adacapeqbrcqr:
	SOLVER_CLASS=SparseNTestAdaCapEquilibratedBroydenCQR  NUM_INSTANCES=1000 env/bin/python -m benchmark.test

# ntest5brcqr: # adaptive, first testing with downsampling
# 	SOLVER_CLASS=SparseNTestEquilibrated5BroydenCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

ntest6brcqr: # adaptive, higher ceil
	SOLVER_CLASS=SparseNTestEquilibrated6BroydenCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

ntest7brcqr: # exponential multiscale step
	SOLVER_CLASS=SparseNTestBroydenMSCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

ntest8brcqr: # exponential multiscale step equilibrated
	SOLVER_CLASS=SparseNTestBroyden2MSCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

# testbrcqrlarge:
# 	SOLVER_CLASS=SparseNTestEquilibrated1BroydenCQR NUM_INSTANCES=1 SIZE_CHOICE=LARGE env/bin/python -m benchmark.test

# hsdenewcqr:
# 	SOLVER_CLASS=HSDENewCQR NUM_INSTANCES=1 env/bin/python -m benchmark.test

eqnewcqr:
	SOLVER_CLASS=EquilibratedNewCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

broydencqr:
	SOLVER_CLASS=BaseBroydenCQR NUM_INSTANCES=10 env/bin/python -m benchmark.test

sparsebroydencqr:
	SOLVER_CLASS=SparseBroydenCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

sparsesoftbroydencqr:
	SOLVER_CLASS=SparseSoftBroydenCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

sparsesoftsqrtbroydencqr:
	SOLVER_CLASS=SparseSoftSqrtBroydenCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

sparsesofticbroydencqr:
	SOLVER_CLASS=SparseSoftInvCorrBroydenCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

sparse2broydencqr:
	SOLVER_CLASS=Sparse2BroydenCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

sparse3broydencqr:
	SOLVER_CLASS=Sparse3BroydenCQR NUM_INSTANCES=10 env/bin/python -m benchmark.test

qr3broydencqr:
	SOLVER_CLASS=QR3BroydenCQR NUM_INSTANCES=100 env/bin/python -m benchmark.test

toybroydencqr:
	SOLVER_CLASS=ToyBroydenCQR NUM_INSTANCES=1 env/bin/python -m benchmark.test

l2eqnewcqr:
	SOLVER_CLASS=L2EquilibratedNewCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

eqlmnewcqr:
	SOLVER_CLASS=EquilibratedLevMarNewCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

posteqlmnewcqr:
	SOLVER_CLASS=PostEquilibratedLevMarNewCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

eqlmnewcqr2:
	SOLVER_CLASS=EquilibratedLevMarNewCQR2Iter NUM_INSTANCES=1000 env/bin/python -m benchmark.test

eqlmnewcqr10:
	SOLVER_CLASS=EquilibratedLevMarNewCQR10Iter NUM_INSTANCES=1000 env/bin/python -m benchmark.test


lmnewcqr:
	SOLVER_CLASS=LevMarNewCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

lmudnewcqr:
	SOLVER_CLASS=LevMarUnitDampNewCQR NUM_INSTANCES=1000 env/bin/python -m benchmark.test

cpr:
	SOLVER_CLASS=SimpleCPR NUM_INSTANCES=10 SIZE_CHOICE=NORMAL env/bin/python -m benchmark.test

eqcpr:
	SOLVER_CLASS=EquilibratedCPR NUM_INSTANCES=10 env/bin/python -m benchmark.test

br2scs:
	SOLVER_CLASS=Broyden2SCS NUM_INSTANCES=10 env/bin/python -m benchmark.test

br2cqr:
	SOLVER_CLASS=Broyden2CQRSCS NUM_INSTANCES=10 env/bin/python -m benchmark.test

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

test:
	BENCHMARK_MODE=TEST env/bin/python -m benchmark.test
