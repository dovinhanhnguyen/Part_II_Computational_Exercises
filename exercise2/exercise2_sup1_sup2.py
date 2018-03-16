
# coding: utf-8

# In[1]:


# Import modules
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


# In[2]:


# Function to return y0dot and y1dot
# We set g=l, Omega = 2/3
def derivatives(y,t,q,F):
    return [y[1], -np.sin(y[0])-q*y[1]+F*np.sin(2*t/3)]


# In[3]:


# Numerical integration starting from different initial conditions
natural_period = 2*np.pi
number_of_oscillations = 50
dt = 0.1

t = np.arange(0.0, natural_period*number_of_oscillations, dt)
y0 = [[0.19999, 0.0], [0.20000, 0.0], [0.20001, 0.0]]
q = 0.5 # damping force
F = 1.2 # driving force
y = [None]*len(y0) # array to store integration values

for i in range(len(y0)):
    y[i] = integrate.odeint(derivatives,y0[i],t,args=(q,F,))


# In[4]:


# Plot angular displacement for different initial conditions
for i in range(len(y0)):
    plt.plot(t,y[i][0:,0])

plt.legend(y0[:],loc='upper left')
plt.xlabel('Time (s)')
plt.ylabel('Angular Displacement (rad)')
plt.savefig('Angular_Displacement_for_Different_Initial_Conditions.pdf')


# In[5]:


# Plot angular velocity for different initial conditions
for i in range(len(y0)):
    plt.plot(t,y[i][0:,1])

plt.legend(y0[:],loc='lower right')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.savefig('Angular_Velocity_for_Different_Initial_Conditions.pdf')

