# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 11:01:21 2022

@author: SENYO ISAAC
"""

from Dynamic_Crowd_Loading import*
from Numerical_Method import Duhamel
from Bridge_1_Pedestrian import Peaks
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

N = 100 #Number of pedestrians that cross the bridge in the time window
window = 10*60 #(s) #Simulation window
buffer = 200 #(s) Additional seconds to allow simulation of response beyond window length (late finishers)
mp = 80 #(kg) Pedestrian mass
G = 9.81*mp #(N) Static weight of pedestrian

#Random variables
tStart = np.random.uniform(low=0.0, high=window, size=N) #Uniformly distributed start times
Vp = np.random.normal(loc=1.3, scale=0.125, size=N) #Normally distributed walking velocities

#Set up the simulation time vector
tMax = window + buffer #(s) Max time
time = np.arange(0, tMax+delT, delT) 

#Initialise containers to hold the individual forces and responses calculated for each pedestrian
crowdForce = np.zeros([N,len(time)])
crowdResponse = np.zeros([N,len(time)])

#For each pedestrian...
for i, n in enumerate(np.arange(N)):    
    vp = Vp[i] #(m/s) Walking velocity
    startTime = tStart[i] #(s) Start time    
    tCross = L/vp #(s) Crossing time
    tEnd = startTime + tCross #(s) Finish time    

    fv = 0.35*vp**3 - 1.59*vp**2 + 2.93*vp #(Hz) Pacing frequency
    DLF = 0.41*(fv-0.95) #Dynamic load factor
    
    timeVector = np.arange(0, tCross+delT, delT) #Time vector for this pedestrian
    Fv = G + abs(G*DLF*np.sin(2*math.pi*(fv/2)*timeVector)) #Static + Dynamic GRF (ignore static component)
    
    xp = vp*timeVector #Position as a function of time
    phi = np.sin(math.pi*xp/L) #Mode shape at pedestrian's location
    Fn = Fv*phi #Modal force experienced by SDoF system
    
    response = Duhamel(timeVector, Fn) #Response calculated using the Duhamel integral function
    
    #Save the GRF and response for this pedestrian at the correct position in the overal simulation records
    iStart = round(startTime/delT) #Index for start time
    crowdForce[i, iStart:iStart+len(Fn)] = Fn
    crowdResponse[i,iStart:iStart+len(Fn)] = response     
    
fig, axes = plt.subplots(figsize=(14,10),nrows=2,ncols=1)

for i in np.arange(len(crowdForce)):    
    axes[0].plot(time,crowdForce[i,:],'-')
    axes[1].plot(time,-crowdResponse[i,:],'-')

#Housekeeping
axes[0].plot([window, window],[0,np.max(crowdForce)],'r--')    
axes[0].plot([window+buffer, window+buffer],[0,np.max(crowdForce)],'r--')  
axes[0].set_xlabel('time (s)')
axes[0].set_ylabel('Force (N)')
axes[0].set_title('Individual Modal forces')
axes[0].set_xlim([0,tMax])
# axes[0].set_xlim([startTime,startTime+tCross])
axes[0].grid()

axes[1].plot([window, window],[0,-np.max(crowdResponse)],'r--')    
axes[1].plot([window+buffer, window+buffer],[0,-np.max(crowdResponse)],'r--')  
axes[1].set_xlabel('time (s)')
axes[1].set_ylabel('Disp (m)')
axes[1].set_title('Individual Modal responses')
axes[1].set_xlim([0,tMax])
# axes[1].set_xlim([startTime,startTime+tCross])
axes[1].grid()

plt.show()

#Sum across rows of crowdForce and crowdResponse
F_Crowd = sum(crowdForce)
Res_crowd = sum(crowdResponse)

peaks, times = Peaks(Res_crowd,time)

fig, axes = plt.subplots(figsize=(14,10),nrows=2,ncols=1) 

axes[0].plot(time,F_Crowd,'-')
axes[1].plot(time,-Res_crowd,'-')
axes[1].plot(times,-peaks,'r-')

axes[0].plot([window, window],[0,max(F_Crowd)],'r--')    
axes[0].plot([window+buffer, window+buffer],[0,max(F_Crowd)],'r--')  
axes[0].set_xlabel('time (s)')
axes[0].set_ylabel('Force (N)')
axes[0].set_title('Crown loading')
axes[0].set_xlim([0,tMax])
axes[0].grid()

axes[1].set_xlabel('time (s)')
axes[1].set_ylabel('Disp (m)')
axes[1].set_title('Modal responses')
axes[1].set_xlim([0,tMax])
axes[1].grid()

plt.show()

animLength = 100 #(sec)
frameRate = 12 #(5,10,20) frames per second (too high and animation slows down)
plotInterval = 1/frameRate #(sec) time between frame plots
dataInterval = int(plotInterval/delT) #Plot moving elements every 'dataInterval-th' point
defScale = 500 #Scale factor on bridge deflection (for visibility)


fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1, figsize=(10, 5)) #Define figure and subplots
gs = gridspec.GridSpec(2,1,height_ratios=[1,1]) #Control subplot layout
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

ax1.set_aspect('equal', adjustable='box') #Set equal scale for axes top subplot

#Set axis limits
ax1.set_xlim([0,L])
ax1.set_ylim([-3, 3])
yLim2 = defScale*max(Res_crowd)
ax2.set_xlim([0,L])
ax2.set_ylim([-yLim2, yLim2])

#Housekeeping
ax1.set_title('Bridge plan view')
ax1.set_xlabel('(m)')
ax1.set_ylabel('(m)')
ax1.grid()

ax2.set_title('Bridge oscillation')
ax2.set_xlabel('(m)')
ax2.set_ylabel(f'Scaled displacement\n x{defScale} (m)')
ax2.grid()

plt.show()

#Define initial state of pedestrians in top plot
topPedList = [] #Initialise an empty list to hold markers representing pedestrians
for i in np.arange(N): 
    yPos = np.random.uniform(low=-2.5, high=2.5, size=1) #Random positions across bridge deck width
    pedTop, = ax1.plot(0,yPos,'o', markeredgecolor='k', markersize=10)
    topPedList.append(pedTop)

#Define initial state of pedestrians in bottom plot
btmPedList = [] 
for i in np.arange(N): 
    ped, = ax2.plot([0,0],[0,0.6*yLim2])
    btmPedList.append(ped)
        
#Define the initial state of the beam in the bottom plot
xVals = np.arange(0,L+1,1) #An array of x-values along the beam
phiVals = np.sin(math.pi*xVals/L) #Corresponding y-values
beamDisp = 0*phiVals #Initial array of displacements along the beam

axisLine, = ax2.plot(xVals,beamDisp,'k') #Add a horizontal beam axis to plot
defLine, = ax2.plot(xVals,beamDisp,'r') #Add initial deflected shape to plot

#Function to animate plot objects
def animate(i):
    frm = int(i*dataInterval) #Index of data for this frame
    simTime = time[frm] #Simulation time for this animation frame
    
    #Update the pedestrian positions (top plot) for the current frame
    for i in np.arange(N):
        if(simTime>tStart[i] and simTime<tStart[i] + L/Vp[i]):
            Pt = topPedList[i]
            pos = (simTime - tStart[i])*Vp[i]
            Pt.set_xdata([pos, pos])
    
    
    #Update the beam deflected shape for the current frame
    defLine.set_data(xVals, -defScale*phiVals*Res_crowd[frm])
    
    #Update the pedestrian positions (bottom plot) for the current frame
    for i in np.arange(N):
        if(simTime>tStart[i] and simTime<tStart[i] + L/Vp[i]):
            Pb = btmPedList[i]
            pos = (simTime - tStart[i])*Vp[i]            
            h = 0.1 + 0.1*crowdForce[i,frm]/max(crowdForce[i,:])            
            Pb.set_data([pos, pos],[0,h])
    
   

#Function to generate the animation
myAnimation = FuncAnimation(fig, 
                            animate, 
                            frames=int(1 + (animLength/plotInterval) ),
                            interval=plotInterval*1000,  #milliseconds
                            blit=True, 
                            repeat=True)

plt.show()
myAnimation.save('Bridge_response.gif')