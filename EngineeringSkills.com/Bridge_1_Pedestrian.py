# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 21:29:08 2022

@author: SENYO ISAAC
"""
from Dynamic_Crowd_Loading import*
from Numerical_Method import Duhamel

xp = vp*time #Pedestrian position as a function of time
phi = np.sin(math.pi*xp/L) #Mode shape at pedestrian's location
Fn = Fv*phi #Modal force experienced by SDoF system

M = 2000 #(kg/m) Mass per unit length]
m = 0.5*M*L #(kg) Modal mass of mode 1
xi = 0.025 #Damping ratio
fn = 2.5 #(Hz) Bridge modal frequency
wn = 2*math.pi*fn #(rads/s) Angular modal frequency
wd = wn*math.sqrt(1-xi**2) #(rads/s) Damped angular modal frequency

response = Duhamel(time, Fn) #Response calculated using the Duhamel integral function

fig = plt.figure() 
axes = fig.add_axes([0.1,0.1,2,1]) 
axes.plot(time,-response, '-')  

#Housekeeping
axes.set_xlabel('time (s)')
axes.set_ylabel('Disp (m)')
axes.set_title('Modal response (static+dynamic)')
axes.set_xlim([0,tMax])
plt.grid()
plt.show()


Fn_static = G*phi #Static component of GRF
Fn_dynamic = abs(G*DLF*np.sin(2*math.pi*(fv/2)*time))*phi #Dynamic component of GRF

response_static = Duhamel(time, Fn_static) #Response due to constant magnitude moving load
response_dynanmic = Duhamel(time, Fn_dynamic) #Response due to footsteps (with static load component removed)

fig, axes = plt.subplots(figsize=(14,10),nrows=2,ncols=1) 
axes[0].plot(time,-response_dynanmic, '-', label='Dynamic')  
axes[0].plot(time,-response_static, 'r-', label='Static') 
axes[0].set_xlabel('time (s)')
axes[0].set_ylabel('Disp (m)')
axes[0].set_title('Modal response (separate components)')
axes[0].legend(loc='lower right')
axes[0].set_xlim([0,tMax])
axes[0].grid()

axes[1].plot(time,-response, '-', label='Combined')  
axes[1].set_xlabel('time (s)')
axes[1].set_ylabel('Disp (m)')
axes[1].set_title('Modal response (combined)')
axes[1].legend(loc='lower right')
axes[1].set_xlim([0,tMax])
axes[1].grid()

plt.show()


def Peaks(disp, time):
    #Initialise containers to hold peaks and their times
    peaks = np.empty([1,0]) 
    times = np.empty([1,0])
    
    #Calculate slopes for each data point
    slopes = np.zeros(len(disp))
    for i, u in enumerate(disp):
        if(i<len(disp)-1):
            slopes[i] = disp[i+1]-disp[i]
    
    #Cycle through all slopes and pick out peaks
    for i, s in enumerate(slopes):
        if (i<len(slopes)-1):
            if(slopes[i+1]<0 and slopes[i]>0):
                peaks = np.append(peaks,disp[i])
                times = np.append(times,time[i])
                
    return [peaks, times]                     

peaks, times = Peaks(response,time)

fig = plt.figure() 
axes = fig.add_axes([0.1,0.1,2,1]) 
axes.plot(time,-response, '-', label='Response')  
axes.plot(times,-peaks,'r-', label='Response envelope')

#Housekeeping
axes.set_xlabel('time (s)')
axes.set_ylabel('Disp (m)')
axes.set_title('Modal response and envelope')
axes.legend(loc='lower right')
axes.set_xlim([0,tMax])
plt.grid()
plt.show()

# Analysing the peaks of the brige response to different masses

k = m*wn**2 #(N/m) Original system stiffness
Masses = [1750, 2000, 2250] #(kg/m) Masses per unit length to test

fig = plt.figure() 
axes = fig.add_axes([0.1,0.1,2,1])

for M in Masses:    
    m = 0.5*M*L #(kg) Modal mass of mode 1
    wn = math.sqrt(k/m)
    wd = wn*math.sqrt(1-xi**2) #(rads/s) Damped angular modal frequency
        
    response = Duhamel(time, Fn) #Response calculated using the Duhamel integral function
    peaks, times = Peaks(response,time)
    
    axes.plot(time,response,'-', label=f'Response M={M}')
    axes.plot(times,-peaks,'-', label=f'Response envelope M={M}')
    
#Housekeeping
axes.set_xlabel('time (s)')
axes.set_ylabel('Disp (m)')
axes.set_title('Modal responses (varying bridge mass)')
axes.legend(loc='lower right')
axes.set_xlim([0,tMax])
plt.grid()
plt.show()