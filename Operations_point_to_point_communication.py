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

    for worker in range(1,size):
        #Send both variables to each worker
        comm.send(a_chunks[worker],dest=worker,tag=0)
        comm.send(b_chunks[worker],dest=worker,tag=1)
    #Set variables for worker 0
    a=a_chunks[0]
    b=b_chunks[0]

else:
    #Receive variables
    a=comm.recv(source=0,tag=0)
    b=comm.recv(source=0,tag=1)

#Perform operation
result=a@b

if rank==0:
    #Calculate and print result
    for worker in range(1,size):
        result+=comm.recv(source=worker,tag=2)
    logger.warning(f'Result: {result}')
    logger.warning(f" Took: { time.time() - start }")
else:
    #Send intermidiate results
    comm.send(result,dest=0,tag=2)