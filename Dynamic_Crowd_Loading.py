# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 21:26:02 2022

@author: SENYO ISAAC
"""
# DEPENDENCIES and DEFAULTS
import math #.....................................Math functionality
import numpy as np #..............................Numpy for working with arrays
import matplotlib.pyplot as plt #.................Plotting functionality 

L = 60 #(m) Bridge span
vp = 1.3 #(m/s) Pedestrian walking velocity
tMax = L/vp #(s) Crossing time
m = 80 #(kg) Pedestrian mass
G = 9.81*m #(N) Static weight of pedestrian

fv = 0.35*vp**3 - 1.59*vp**2 + 2.93*vp #(Hz) Pacing frequency
DLF = 0.41*(fv-0.95) #Dynamic load factor
print(f"- The DLF = {round(DLF,3)} and the pacing frequency is {round(fv,2)} Hz ({round(fv,2)} steps per second)")
print(f"- Duration of a single step is {round(1/fv,2)} seconds")

delT = 0.005 #(s) Time-step
time = np.arange(0, tMax+delT, delT) #Time vector
Fv = G + abs(G*DLF*np.sin(2*math.pi*(fv/2)*time)) #Static + Dynamic GRF

fig = plt.figure() 
axes = fig.add_axes([0.1,0.1,2,1]) 
axes.plot(time,Fv, '-', label='GRF')  

#Housekeeping
axes.set_xlabel('time (s)')
axes.set_ylabel('Force (N)')
axes.set_title('Vertical ground reaction force')
axes.legend(loc='lower right')
axes.set_xlim([0,5])
plt.grid()
plt.show()