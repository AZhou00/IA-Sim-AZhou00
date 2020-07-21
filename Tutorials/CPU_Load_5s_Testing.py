from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
rank = comm.rank
for i in range(500000000):
    if i>=-1:
        pass
print('rank = ',rank,'done')
