import h5py
import numpy as np
from numpy.random import (
    Generator,
    PCG64,
    PCG64DXSM,
    MT19937,
    Philox,
    SFC64
)

g = [PCG64, PCG64DXSM, MT19937, Philox, SFC64]
length = 512 * 512


def create_split(posl_amount, seed, name):
    total_amount = 5 * posl_amount
    generators = [Generator(gen(seed)) for gen in g]
    with h5py.File(name, "w") as f:
        dset_x = f.create_dataset("X", shape=(total_amount, length), dtype=np.uint8)
        dset_y = f.create_dataset("y", shape=(total_amount,), dtype=np.int64)

        row_i = 0
        for i, rng in enumerate(generators):
            for _ in range(posl_amount):
                dset_x[row_i] = rng.integers(0, 256, size=length, dtype=np.uint8)
                dset_y[row_i] = i
                row_i += 1

create_split(1000, seed=16, name="train_dataset.hdf5")
create_split(100, seed=32, name="test_dataset.hdf5")
