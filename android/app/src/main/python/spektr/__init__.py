"""spektr — a terminal spectrum analyser for whatever your speakers are doing."""

import os as _os

# One thread for NumPy's linear algebra, set before NumPy is first imported --
# which is why it is here, the first thing any spektr import runs. The OpenBLAS
# bundled with NumPy starts a worker per core and commits a buffer for each: on
# a sixteen-core machine, importing NumPy alone committed 499 MB of private
# memory against 17 MB with one thread, and the idle workers spin. spektr's
# matrix work is a handful of twelve-by-twelve products that gain nothing from
# threads. ``setdefault``, so anyone who has set these themselves keeps theirs.
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")
del _var

__version__ = "0.6.0"

__all__ = ["__version__"]
