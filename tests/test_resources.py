"""What spektr costs just by being open."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env_after_import(extra: dict | None = None) -> str:
    env = {k: v for k, v in os.environ.items()
           if k not in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")}
    env.update(extra or {})
    code = ("import spektr, numpy, os; "
            "print(os.environ['OPENBLAS_NUM_THREADS'])")
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, text=True, check=True).stdout.strip()


def test_numpy_gets_one_blas_thread():
    """OpenBLAS commits a buffer per core: 499 MB of private memory on a
    sixteen-core machine for importing NumPy, against 17 MB with one thread.
    spektr's linear algebra is a few twelve-by-twelve products."""
    assert _env_after_import() == "1"


def test_a_setting_of_the_user_s_own_is_kept():
    assert _env_after_import({"OPENBLAS_NUM_THREADS": "4"}) == "4"


def test_the_analyser_sleeps_until_audio_arrives():
    """It used to poll every millisecond, a thousand wakeups a second in
    silence as much as in music. A push wakes it instead."""
    import threading
    import time

    from spektr.capture import RingBuffer

    ring = RingBuffer(4096)
    woke = []

    def waiter():
        t0 = time.perf_counter()
        ring.wait(2.0)
        woke.append(time.perf_counter() - t0)

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.05)
    import numpy as np

    ring.push(np.zeros((256, 2), dtype=np.float32))
    t.join(3.0)
    assert woke and woke[0] < 1.0, "the push did not wake the waiter"
