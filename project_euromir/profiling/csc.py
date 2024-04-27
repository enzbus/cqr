import project_euromir as lib
import time
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

m = 1000
n = 1000
tries = 100
timers = np.empty(tries, dtype=float)

meantimes = []
stdtimes = []
densities = np.linspace(.01, 1, 100)
inp = np.random.randn(n)
out = np.random.randn(m)
mult = 1.

for density in densities:

    mat = sp.random(m=m, n=n, dtype=float, density=density).tocsc()

    for seed in range(tries):
        np.random.seed(seed)
        inp[:] = np.random.randn(n)
        s = time.time()
        lib.add_csc_matvec(
            n=n, col_pointers=mat.indptr, row_indexes=mat.indices,
            mat_elements=mat.data, input=inp, output=out, mult=mult)
        timers[seed] = time.time() - s
    print('timer CSC', np.median(timers))
    meantimes.append(np.mean(timers))
    stdtimes.append(np.std(timers))

## LINEAR FIT USING SCIPY W/ERROR PROPAGATION

def linear_f(x,a,b):
    return a*x+b

(a, b), Sigma = curve_fit(
    linear_f,xdata=densities,ydata=meantimes,sigma=stdtimes,
    absolute_sigma=True)
stda, stdb = np.sqrt(np.diag(Sigma))
print(f'INTERCEPT (s): {a:.3e} plusminus {stda:.3e}')
print(f'SLOPE (s/density): {b:.3e} plusminus {stdb:.3e} ')

plt.errorbar(x=densities, y=meantimes, yerr=stdtimes)
plt.plot(densities, linear_f(densities, a, b))
plt.title(f'matvec of a {m}x{n} matrix')
plt.ylim(
    b + np.min(densities)*a - 2*np.median(stdtimes), 
    b + np.max(densities)*a + 2*np.median(stdtimes)
    )
plt.ylabel(f'median time out of {tries}')
plt.xlabel('density')
plt.show()

