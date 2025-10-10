# Copyright 2024 Enzo Busseti
#
# This file is part of CQR, the Conic QR Solver.
#
# CQR is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# CQR is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# CQR. If not, see <https://www.gnu.org/licenses/>.
"""Usage

python plot.py SimpleSCS # plot one solver prototype
"""

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    raise SyntaxError("Specify one solver name.")

solver = sys.argv[1]

for result_file in list(Path('results/').glob(f"{solver}*.npy.gz")):
    program_name = "_".join(result_file.stem.split('.')[0].split('_')[1:])
    plt.figure()
    with gzip.open(result_file, "r") as f:
        sol_quals = pd.DataFrame(np.load(file=f))
    sol_quals = sol_quals.fillna(np.inf)
    program_name = "_".join(result_file.stem.split('.')[0].split('_')[1:])
    plt.semilogy(sol_quals.quantile(.5, axis=1).values, label=solver + ", 50%")
    # plt.semilogy(sol_quals.quantile(.75, axis=1))
    plt.semilogy(sol_quals.quantile(.95, axis=1).values, label=solver + ", 95%")
    plt.semilogy(sol_quals.quantile(.99, axis=1).values, label=solver + ", 99%")
    plt.semilogy(sol_quals.max(axis=1).values, label=solver + ", max")
    plt.legend()
    plt.title(f"{solver}, {program_name}, worst seed {sol_quals.iloc[-1].argmax()}")

plt.show()        # import matplotlib.pyplot as plt

