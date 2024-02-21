#mpiexec -n 4 python 

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

if rank==0:
    #Create data
    X = np.arange(0, 1, 0.01)
    Y = X + np.random.normal(0, 0.2, len(X))

    #Initialize parameters
    params=np.array([-0.5, 0.2])
    initial_params=params
    lr=0.006
    iterations=10000
    keep_going=True
    advancement=True
    epsilon=0.0001
    history=[]
    history.append(params)

    #Split data
    x_chunk=np.array_split(X, size )
    y_chunk=np.array_split(Y, size )
else:
    #Initialize variables
    x_chunk=None
    y_chunk=None
    params=None
    keep_going=None

#Scatter data to workers
x_chunk=comm.scatter(x_chunk,root=0)
y_chunk=comm.scatter(y_chunk,root=0)

params=comm.bcast(params,root=0)
keep_going=comm.bcast(keep_going, root=0)

#Define function
def y_hat(x, theta1, theta2):
    return theta1*x+theta2

#If stopping criteria is not met
while keep_going:
    #Data dependent step according to gradient descent
    grad1=np.sum(x_chunk*-2*(y_chunk-y_hat(x_chunk,params[0],params[1])))
    grad2=np.sum(-2*(y_chunk-y_hat(x_chunk,params[0],params[1])))

    #Get sum of calculations by each worker
    comm.reduce(grad1,op=MPI.SUM,root=0)
    comm.reduce(grad2,op=MPI.SUM,root=0)
    
    if rank==0:
        #Update step according to gradient descent
        new1=params[0]-lr*grad1
        new2=params[1]-lr*grad2
    
        #Stop if there is no improvement
        if abs(new1-params[0])<epsilon and abs(new2-params[1])<epsilon:
            advancement=False
        #Stop if max iterations
        if advancement==False or len(history)>iterations:
            keep_going=False
        else:
            #Update params
            params=np.array([new1, new2]) 
            history.append(params)

    #Broadcast new parameters and stopping criterion
    params=comm.bcast(params,root=0)
    keep_going=comm.bcast(keep_going, root=0)

#Plot results
if rank==0:
    print(f'Used {initial_params} as starting point')
    print(f'Stopped after {len(history)} iterations')
    plt.scatter(X,Y,c='r')
    plt.plot(X,y_hat(X,params[0],params[1]))
    plt.title(f'Theta 1:{round(params[0],3)} - Theta 2:{round(params[1],3)}')
    plt.show()
