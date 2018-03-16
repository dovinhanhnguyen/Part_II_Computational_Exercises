
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


# Numerical integration for different values of damping
natural_period = 2*np.pi
number_of_oscillations = 10
dt = 0.1

t = np.arange(0.0, natural_period*number_of_oscillations, dt)
y0 = [0.01, 0.0]
q = [0,0.5,1,5,10] # damping force
F = 0.0 # driving force
y = [None]*len(q) # array to store integration values

for i in range(len(q)):
    y[i] = integrate.odeint(derivatives,y0,t,args=(q[i],F,))


# In[4]:


# Plot angular displacement with damping
for i in range(len(q)):
    plt.plot(t,y[i][0:,0])

plt.legend(q,loc='lower right')
plt.xlabel('Time (s)')
plt.ylabel('Angular Displacement (rad)')
plt.savefig('Angular_Displacement_with_Damping.pdf')


# In[5]:


# Plot angular velocity with damping
for i in range(len(q)):
    plt.plot(t,y[i][0:,1])

plt.legend(q,loc='lower right')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.savefig('Angular_Velocity_with_Damping.pdf')


# In[6]:


# Numerical integration for different values of driving force (with damping)
natural_period = 2*np.pi
number_of_oscillations = 10
dt = 0.1

t = np.arange(0.0, natural_period*number_of_oscillations, dt)
y0 = [0.01, 0.0]
q = 0.5 # damping force
F = [0, 0.5, 1.2, 1.44, 1.465] # driving force
y = [None]*len(F) # array to store integration values

for i in range(len(F)):
    y[i] = integrate.odeint(derivatives,y0,t,args=(q,F[i],))


# In[7]:


# Plot some result with both driving and damping forces
for i in range(len(F)):
    plt.plot(t,y[i][0:,0])

plt.legend(F,loc='lower right')
plt.xlabel('Time (s)')
plt.ylabel('Angular Displacement (rad)')
plt.savefig('Angular_Displacement_with_Damping_and_Driving.pdf')


# In[8]:


# Plot angular velocity with both driving and damping forces
for i in range(len(F)):
    plt.plot(t,y[i][0:,1])

plt.legend(F,loc='lower right')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.savefig('Angular_Velocity_with_Damping_and_Driving.pdf')

