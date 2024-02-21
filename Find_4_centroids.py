#mpiexec -n 4 python Ex2-3.py

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

k=4
dims=2

if rank==0:
    #Load data
    data=pd.read_csv('cluster_data.csv', header=0, index_col=0)
    
    #Split data
    chunks=np.array_split(data, size )

    #Use weighted average to find centroid
    def find_new_centroids(means,ns):
        return np.sum(means*ns/np.sum(ns))
    
    #Initialize variables
    centroids=np.random.random([k,dims])
    keep_going=True
    last_step=0


else:
    #Initialize variables
    chunks=None
    keep_going=None
    centroids=None

#Scatter data
chunks=comm.scatter(chunks,root=0)
#Broadcast variables
keep_going = comm.bcast(keep_going, root=0)
centroids=comm.bcast(centroids, root=0)

#Calculate euclidean distance of each point to each of the centroids
def calculate_distances(data,centroids):
    distances=np.empty([len(data),len(centroids)])
    for i in range(len(centroids)):
        distances[:,i]=np.linalg.norm(data-np.tile(centroids[i], [len(data),1]),axis=1)
    return distances

while keep_going:
    #Classify local data
    distances=calculate_distances(chunks,centroids)
    classification=np.argmin(distances,axis=1)

    #Find mean and weight for each class
    n=np.empty(k)
    mean_x=np.empty(k)
    mean_y=np.empty(k)

    for i in range(k):
        class_data=chunks.iloc[classification==i]
        n[i]=len(class_data)
        if n[i]>0:
            mean_x[i]=class_data['x'].mean()
            mean_y[i]=class_data['y'].mean()
        else:
            mean_x[i]=0
            mean_y[i]=0
        
        

    #Gather data at rank 0
    full_mean_x = comm.gather(mean_x, root=0) 
    full_mean_y = comm.gather(mean_y, root=0) 
    ns = comm.gather(n,root=0)

    if rank ==0:
        #Format data
        full_mean_x, full_mean_y = np.concatenate(full_mean_x).ravel(), np.concatenate(full_mean_y).ravel()
        full_mean_x, full_mean_y = np.reshape(full_mean_x,(k,size)), np.reshape(full_mean_y,(k,size))

        ns = np.concatenate(ns).ravel()
        ns=np.reshape(ns,(k,size))
  
        #Recalculate centroids
        new_centroids=np.empty([k,dims])
        for i in range(k):
            new_centroids[i,0]=find_new_centroids(np.array(full_mean_x[i,:]),np.array(ns[i,:]))
            new_centroids[i,1]=find_new_centroids(np.array(full_mean_y[i,:]),np.array(ns[i,:]))
        
        new_centroids[np.isnan(new_centroids)] = 0
        #Log step advancement
        step_adv=abs(np.sum(new_centroids-centroids))
        logger.warning(step_adv)
        
        #Check stopping condition
        if abs(np.sum(new_centroids-centroids))<0.0001 or step_adv==last_step:
            keep_going=False
        else:
            centroids=new_centroids
        last_step=step_adv
            
    #Broadcast stopping condition and new centroids
    keep_going = comm.bcast(keep_going, root=0)
    centroids=comm.bcast(centroids, root=0)

if rank==0:
    #Log results
    logger.warning(f"\n Centroids: \n{ centroids}")
    logger.warning(f" Took: { time.time() - start }")

    #Classify all data
    distances=calculate_distances(data,centroids)
    classification=np.argmin(distances,axis=1)

    logger.warning(f"K1: { len(classification[ classification == 0])} points")
    logger.warning(f"K2: { len(classification[ classification == 1])} points")
    logger.warning(f"K3: { len(classification[ classification == 2])} points")
    logger.warning(f"K4: { len(classification[ classification == 3])} points")

    #Show and save results
    plt.scatter(data['x'],data['y'], alpha=0.7, c=classification)
    plt.savefig('K_means.png')
    plt.show()

