#mpiexec -n 4 python Ex2-2.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import logging
import time

start = time.time()

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

logging.basicConfig ( format = "%(name)s: %(message)s")
logger = logging . getLogger ( " MyLogger " + str(rank) )

if rank==0:
    #Load data
    data=pd.read_csv('cluster_data.csv', header=0, index_col=0)
    #Split data
    chunks=np.array_split(data, size )

    #Use weighted average to find centroid
    def find_centroid(means,ns):
        return np.sum(means*ns/np.sum(ns))

else:
    #Initialize variable
    chunks=None

#Scatter data
chunks=comm.scatter(chunks,root=0)

#Find local mean and length of local data
mean_x=np.mean(chunks['x'])
mean_y=np.mean(chunks['y'])
n=len(chunks)

#Gather data from workers
x_means = comm.gather(mean_x, root=0)
y_means = comm.gather(mean_y, root=0)
ns = comm.gather(n,root=0)

if rank==0:
    #Print results
    logger.warning(f'Centroid: [ {find_centroid(np.array(x_means),np.array(ns)) , find_centroid(np.array(y_means),np.array(ns))} ]')
    logger.warning(f" Took: { time.time() - start }")


