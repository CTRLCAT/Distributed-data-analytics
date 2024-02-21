#python Ex2-1.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Load data
data=pd.read_csv('cluster_data.csv', header=0, index_col=0)

#Create plot
plt.scatter(data['x'],data['y'], alpha=0.7)

#Save figure
plt.savefig('Scattered_data.png')

#Show result
plt.show()


