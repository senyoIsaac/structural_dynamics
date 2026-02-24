# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 21:03:50 2022

@author: SENYO ISAAC
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym




m = 1000 #(kg) Mass
xi = 0.05 # Damping ratio
f = 1.5 #(Hz) Natural frequency
wn = 2*math.pi*f #(rads/s) Angular natural frequency
wd = wn*math.sqrt(1-xi**2) #(rads/s) Damped angular natural frequency


tMax = 20 #(s) Max time
delT = 0.01 #(s) Time-step
time = np.arange(0, tMax+delT, delT) #Time vector

f_Force = 1 #(Hz) Forcing frequency
wf = 2*math.pi*f_Force #(rads/s) Angular forcing frequency
P=100 #(N) Force amplitude
force = P*np.sin(wf*time) #Force vector


def Duhamel(T, F):
    #Initialise a container of zeros to hold the displacement values
    U = np.zeros(len(T))
    
    #Initialise values for the cumulative sum used to calculate A and B at each time-step
    ACum_i=0 
    BCum_i=0
    
    #Cycle through the time vector and evaluate the response at each time point
    for i, t in enumerate(T):
        #Only start calculating on the second iteration (need two values for trapezoidal area calculation)
        if i>0:     
            #Calculate A[i] - equation 4
            y_i = math.e**(xi*wn*T[i]) * F[i] * math.cos(wd*T[i]) #Value of integrand at current time-step
            y_im1 = math.e**(xi*wn*T[i-1]) * F[i-1] * math.cos(wd*T[i-1]) #Value of integrand at previous time-step
            Area_i = 0.5*delT*(y_i+y_im1) #Area of trapezoid
            ACum_i += Area_i #Cumulative area from t=0 to current time
            A_i = (1/(m*wd))*ACum_i #Value of A for the current time-step
            
            #Calculate B[i] - equation 5 (same notes as for A above)
            y_i = math.e**(xi*wn*T[i]) * F[i] * math.sin(wd*T[i])
            y_im1 = math.e**(xi*wn*T[i-1]) * F[i-1] * math.sin(wd*T[i-1])
            Area_i = 0.5*delT*(y_i+y_im1)
            BCum_i += Area_i
            B_i = (1/(m*wd))*BCum_i                
                  
            #Calculate the response - equation 3
            U[i] = A_i*math.e**(-xi*wn*T[i])*math.sin(wd*T[i]) - B_i * math.e**(-xi*wn*T[i])*math.cos(wd*T[i])             
            
    return U



response = Duhamel(time, force) #Response calculated using the Duhamel integral function

#Initialise a fig to plot onto
fig = plt.figure() 
axes = fig.add_axes([0.1,0.1,2,1]) 
axes.plot(time,response)    

#Housekeeping
axes.set_xlabel('time (s)')
axes.set_ylabel('Displacement (m)')
axes.set_title('SDoF system response to harmonic loading')
axes.set_xlim([0,tMax])
plt.grid()
plt.show()