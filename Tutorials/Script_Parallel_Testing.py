from mpi4py import MPI
comm = MPI.COMM_WORLD #can pull name, rank, size etc. from this. This object is the barebone basic of MPI
print('Hi, my rank is:', comm.rank)
if comm.rank == 1: #if this i the node 1
    print('doing the task for rank(node) 1')
elif comm.rank == 0: 
    print('doing the task for rank(node) 0')
elif comm.rank == 2:
    print('doing the task for rank(node) 2')

print('finishing node %i \n\n'%comm.rank)
