#mpiexec -n 4 python 

from mpi4py import MPI
import logging
import numpy as np
from datetime import datetime

start = datetime . now ()

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

logging.basicConfig ( format = "%(name)s: %(message)s")
logger = logging . getLogger ( " MyLogger " + str(rank) )
max_num=10000

if rank==0:
    #Initialize variable
    my_list = list ( range (max_num) )
    np.random.shuffle( my_list )

    #Separate into chunks
    chunks=np.array_split(my_list, size )
    keep_going=True

else:
    #Initilize for other workers
    chunks=None
    keep_going=None

#Create stopping variable
keep_going = comm.bcast(keep_going, root=0)
#Scatter data
chunks=comm.scatter(chunks,root=0)
#Perform operation and select min value
chunks=np.sort(chunks)
first=chunks[0]

while keep_going:
    #Show next number
    next_number=comm.reduce(first,op=MPI.MIN,root=0) 
    #Stopping condition
    if next_number==max_num:
        keep_going=False

    #Tell workers number used
    num_used = comm.bcast(next_number, root=0)

    if rank==0:
        #Print number
        logger.warning(f'{next_number}')
    
    if first==num_used and len(chunks)>0:
        #Move data in chunk
        chunks=np.delete(chunks, 0)
    
    #Select new first in chunk
    if len(chunks)>0:
        first=chunks[0]
    else:
        first=max_num

    keep_going = comm.bcast(keep_going, root=0)