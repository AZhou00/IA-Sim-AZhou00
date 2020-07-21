from mpi4py import MPI
import numpy as np
import os
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print('this is rank', rank)
print(os.path)
if not os.path.exists('/home/azhou/IA-Sim-AZhou00/IA_Numeric_Output/tests'):
    os.makedirs('/home/azhou/IA-Sim-AZhou00/IA_Numeric_Output/tests')
filename = os.path.join('/home/azhou/IA-Sim-AZhou00/IA_Numeric_Output/tests', 'test_array%i'%rank)
file = open(filename, "wb")
np.save(file, np.array([rank,rank*2,rank*3]))
file.close
