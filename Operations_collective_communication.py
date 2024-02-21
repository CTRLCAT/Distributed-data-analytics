#mpiexec -n 4 python 

from mpi4py import MPI
import logging
import numpy as np
import time

try: 
    start
except NameError:
    start = time.time()

logging.basicConfig ( format = "%(name)s: %(message)s")
logger = logging.getLogger ( " MyLogger " )

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
data_size=6

if rank==0:
    #Create variables
    a=[i for i in range(data_size)]
    b=[i for i in range(data_size,data_size*2)]

    #Separate into chunks
    a_chunks=np.array_split (a, size )
    b_chunks=np.array_split (b, size )

else:
    #Initialize for other workers
    a_chunks=None
    b_chunks=None

#Scatter both variables
a_chunks=comm.scatter(a_chunks,root=0)
b_chunks=comm.scatter(b_chunks,root=0)

#Perform operation
result=a_chunks@b_chunks

#Calculate final result
results = comm.reduce(result, op=MPI.SUM, root =0)

if rank==0:
    logger.warning(f'Result: {results}')
    logger.warning(f" Took: { time.time() - start }")