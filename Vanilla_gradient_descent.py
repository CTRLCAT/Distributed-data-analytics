#python 
import numpy as np
import matplotlib.pyplot as plt

#Define functions
def y_hat(x):
    return np.square(x)

def gradient(x):
    return 2*x

#Define parameters
iterations=20
lr=0.1
x=3.5
epsilon=0.00001

step_log=[]
step_log.append(x)

#Stop at max iterations
for i in range(iterations):
    #Calculate new x according to gradient descent
    new_x=x-lr*gradient(x)
    #Check if there is no improvement
    if abs(new_x-x)<epsilon:
        break
    else:
        #Update x and save step
        x=new_x
        step_log.append(x)

#Plot results
x_function=np.arange(-4,4.2,0.2)

plt.plot(x_function,y_hat(x_function))
plt.scatter(step_log,y_hat(step_log),c='r')
plt.title(f'Function: x^2')
plt.show()
