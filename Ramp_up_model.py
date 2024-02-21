#python 

import numpy as np
import matplotlib.pyplot as plt

#Create data
X = np.arange(0, 1, 0.01)
Y = X + np.random.normal(0, 0.2, len(X))

#Define function
def y_hat(x, theta1, theta2):
    return theta1*x+theta2

#Initialize parameters
theta1=1
theta2=0

#Plot data and function
plt.scatter(X,Y,c='r')
plt.plot(X,y_hat(X,theta1,theta2))
plt.title(f'Theta 1:{theta1} - Theta 2:{theta2}')
plt.show()

#Inintialize parameters
theta1=-0.5
theta2=0.2
lr=0.006
iterations=20

#Do for n iterations
for i in range(iterations):
    #Update both parameters according to gradient descent
    theta1=theta1-lr*np.sum(X*-2*(Y-y_hat(X,theta1,theta2)))
    theta2=theta2-lr*np.sum(-2*(Y-y_hat(X,theta1,theta2)))

#Plot results
plt.scatter(X,Y,c='r')
plt.plot(X,y_hat(X,theta1,theta2))
plt.title(f'Theta 1:{round(theta1,3)} - Theta 2:{round(theta2,3)}')
plt.show()

