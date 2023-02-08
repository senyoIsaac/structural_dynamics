# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 21:01:21 2022

@author: SENYO ISAAC
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
sym.init_printing()

m = sym.Symbol('m')
w = sym.Symbol('w')
Po = sym.Symbol('Po')
t1 = sym.Symbol('t1')
tau = sym.Symbol('tau')
t = sym.Symbol('t')

f= tau * sym.sin(w*t-w*tau)

defInt= sym.integrate(f,(tau,0,t))
sym.simplify(defInt)

Po = 1000
t1 = 10
delT = 0.1
t = np.arange(0,t1+delT,delT) #Time vector
m = 20 # Mass of System
periodRange = [0.3,0.4,0.5]

#Initialize a figure to plot onto

fig = plt.figure()
axes = fig.add_axes([0.1,0.1,2,1])

for pr in periodRange:
    T = pr*t1
    wn = 2*math.pi/T
    k = m*wn**2
    u = (Po/k)*((t/t1)-((np.sin(wn*t))/(wn*t1)))
    axes.plot(t/t1,u/(Po/k),label=f'T={pr}t1')
    
#Housekeeping

axes.set_xlabel('t/t1')
axes.set_ylabel('Displcement ratio')
axes.set_title('SDoF system response to ramp loading')
axes.legend(loc='upper left')
axes.set_xlim([0,1])
plt.grid()
plt.show