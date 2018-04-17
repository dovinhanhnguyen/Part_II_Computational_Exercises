
# coding: utf-8

# In[ ]:


# Import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import time


# In[ ]:


# Class representing a 2D Ising model
class Ising:
    def __init__(self, N, J, H):
        self.lattice = np.random.choice([-1,1], size=(N,N)) # initial state
        self.N = N # dimension
        self.J = np.abs(J) # interaction energy, non-negative by definition
        self.H = H # external field


# In[ ]:


# Function to calculate energy change when spin(i,j) is flipped, assuming periodic boundary conditions
def energy_change_of_flip(state, pos_i, pos_j):
    return 2*state.lattice[pos_i,pos_j]*(state.J*(state.lattice[pos_i,(pos_j+1)%state.N] + state.lattice[(pos_i-1)%state.N,pos_j] + state.lattice[(pos_i+1)%state.N,pos_j] + state.lattice[pos_i,(pos_j-1)%state.N])+state.H)


# In[ ]:


# Monte Carlo sweep using Metropolis algorithm
def monte_carlo_sweep(state, temperature):
    for i in range(len(state.lattice)**2):
        # pick a spin randomly and calculate energy change if it's flipped
        pos_i = np.random.randint(0,len(state.lattice))
        pos_j = np.random.randint(0,len(state.lattice))
        delta_E = energy_change_of_flip(state,pos_i,pos_j)
        
        # to flip or not to flip?
        if delta_E < 0 or np.exp(-delta_E/temperature) > np.random.random():
            state.lattice[pos_i,pos_j] *= -1


# In[ ]:


# Function to calculate magnetisation
def calculate_magnetisation(state):
    return np.sum(state.lattice)


# In[ ]:


# Function to calculate energy
def calculate_energy(state):
    interaction_energy = 0.0
    for pos_i in range(len(state.lattice)):
        for pos_j in range(len(state.lattice)):
            interaction_energy += -state.J*state.lattice[pos_i,pos_j]*(state.lattice[pos_i,(pos_j+1)%state.N] + state.lattice[(pos_i-1)%state.N,pos_j] + state.lattice[(pos_i+1)%state.N,pos_j] + state.lattice[pos_i,(pos_j-1)%state.N])
    interaction_energy *= 0.5 # avoid double counting
    
    magnetisation_energy = -state.H*calculate_magnetisation(state)
    
    return interaction_energy+magnetisation_energy


# In[ ]:


# Function to normalize data by number of lattice sites
def normalize_data(data, N):
    return data/(N**2)


# In[ ]:


# Metropolis algorithm
def Metropolis_algorithm(N_range, J, H, T_range, no_of_sweeps):
    # arrays to store magnetisation and energy time series
    M = np.zeros((len(N_range),len(T_range),no_of_sweeps))
    E = np.zeros((len(N_range),len(T_range),no_of_sweeps))
    M_normalized = np.zeros((len(N_range),len(T_range),no_of_sweeps))
    E_normalized = np.zeros((len(N_range),len(T_range),no_of_sweeps))
    run_time = np.zeros((len(N_range),len(T_range)))
    
    # Monte Carlo sweeps
    for N_index in range(len(N_range)):
        for T_index in range(len(T_range)):
            state = Ising(N_range[N_index],J,H)
            
            t0 = time.time()
            for sweep in range(no_of_sweeps): # run!
                monte_carlo_sweep(state,T_range[T_index])
                
                M[N_index,T_index,sweep] = calculate_magnetisation(state)
                E[N_index,T_index,sweep] = calculate_energy(state)
                M_normalized[N_index,T_index,sweep] = normalize_data(M[N_index,T_index,sweep],state.N)
                E_normalized[N_index,T_index,sweep] = normalize_data(E[N_index,T_index,sweep],state.N)
                
                if sweep%1000 == 0: # output checkpoints
                    print('N = {0}, T = {1}, sweep = {2}'.format(N_range[N_index],T_range[T_index],sweep))
            t1 = time.time()
            run_time[N_index,T_index] = t1-t0
    
    return M, E, M_normalized, E_normalized, run_time


# In[ ]:


# Function to plot evolution of data over time
def plot_time_series(data, N_range, N_index_range, T_range, T_index_range, start_point, stop_point, xname, yname):
    for N_index in N_index_range:
        for T_index in T_index_range:
            plt.plot(range(start_point,stop_point),data[N_index,T_index,start_point:stop_point],label='N = {0}, T = {1:.3f}'.format(N_range[N_index],T_range[T_index]))
        plt.legend(loc='upper right')
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.savefig('plots/{0} vs {1} N = {2} from {3} to {4}.pdf'.format(yname,xname,N_range[N_index],start_point,stop_point))
        plt.figure()


# In[ ]:


# Function to calculate the average of a thermodynamic variable
def calculate_thermodynamic_variable(data, N_range, T_range, no_of_equilibrating_sweeps):
    var = np.zeros((len(N_range),len(T_range)))
    
    for N_index in range(len(N_range)):
        for T_index in range(len(T_range)):
            var[N_index,T_index] = np.sum(data[N_index,T_index,no_of_equilibrating_sweeps:])/len(data[N_index,T_index,no_of_equilibrating_sweeps:])
        var[N_index,:] = normalize_data(var[N_index,:],N_range[N_index])
    
    return var


# In[ ]:


# Function to calculate the average of a derivative thermodynamic variable
def calculate_derivative_thermodynamic_average(data, N_range, T_range, power_of_T, no_of_equilibrating_sweeps):
    var = np.zeros((len(N_range),len(T_range)))
    
    for N_index in range(len(N_range)):
        for T_index in range(len(T_range)):
            ave = np.sum(data[N_index,T_index,no_of_equilibrating_sweeps:])/len(data[N_index,T_index,no_of_equilibrating_sweeps:])
            squared_ave = np.sum(data[N_index,T_index,no_of_equilibrating_sweeps:]**2)/len(data[N_index,T_index,no_of_equilibrating_sweeps:])
            var[N_index,T_index] = (squared_ave-ave**2)/(T_range[T_index]**power_of_T)
        var[N_index,:] = normalize_data(var[N_index,:],N_range[N_index])
    
    return var


# In[ ]:


# Function to plot thermodynamic variable against temperature
def plot_temperature_dependence(data, N_range, N_index_range, T_range, T_index_range, xname, yname):
    for N_index in N_index_range:
        plt.plot(T_range[T_index_range[0]:(T_index_range[-1]+1)],data[N_index,T_index_range[0]:(T_index_range[-1]+1)],'-o',label='N = {0}'.format(N_range[N_index]))
        # +1 to include last element
    plt.legend(loc='best')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig('plots/{0} vs {1} from {2:.3f} to {3:.3f}.pdf'.format(yname,xname,T_range[T_index_range[0]],T_range[T_index_range[-1]]))
    plt.figure()


# In[ ]:


# Function to calculate the autocorrelation of a variable
def calculate_autocovariation(data, N_range, T_range, no_of_sweeps, no_of_equilibrating_sweeps):
    no_of_sampling_sweeps = no_of_sweeps-no_of_equilibrating_sweeps
    autocovariation = np.zeros((len(N_range),len(T_range),no_of_sampling_sweeps))
    
    for N_index in range(len(N_range)):
        for T_index in range(len(T_range)):
            ave = np.sum(data[N_index,T_index,:])/no_of_sampling_sweeps
            for tau in range(no_of_sampling_sweeps):
                autocorr = 0.0
                for i in range(no_of_equilibrating_sweeps,no_of_sweeps-tau):
                    autocorr += (data[N_index,T_index,i]-ave)*(data[N_index,T_index,i+tau]-ave)
                autocovariation[N_index,T_index,tau] = autocorr
            if autocovariation[N_index,T_index,0] == 0:
                autocovariation[N_index,T_index,:] = np.ones(no_of_sampling_sweeps)
            else:
                autocovariation[N_index,T_index,:] = autocovariation[N_index,T_index,:]/autocovariation[N_index,T_index,0]
    
    return autocovariation


# In[ ]:


# Function of magnetisation near Tc
def shape_function(x, a, Tc, b):
        return a*(((Tc-x)/Tc)**b)


# In[ ]:


# Function to fit data
def data_fitting(data, N_range, N_index_range, T_range, T_index_range, guess):
    x_data = np.zeros((len(N_index_range),len(T_index_range)))
    y_data = np.zeros((len(N_index_range),len(T_index_range))) # data sliced by N_index_range and T_index_range
    for N_index in range(len(N_index_range)):
        for T_index in range(len(T_index_range)):
            x_data[N_index,T_index] = T_range[T_index_range[T_index]]
            y_data[N_index,T_index] = data[N_index_range[N_index],T_index_range[T_index]]
    
    params = np.zeros((len(N_index_range),3))
    errs = np.zeros((len(N_index_range),3))
    
    for N_index in range(len(N_index_range)):
        popt, pcov = optimize.curve_fit(shape_function,x_data[N_index,:],y_data[N_index,:],guess)
        params[N_index,:] = popt
        errs[N_index,:] = pcov.diagonal()
    
    return params, errs


# In[ ]:


# Set parameters and run the experiment!!!
Ns = [10]
J = 1.0
H = 0.0
Ts = np.linspace(1.8,2.26,40)
sweeps = 12000
equilibrating_sweeps = 2000

magnetisation, energy, magnetisation_per_site, energy_per_site, run_time = Metropolis_algorithm(Ns,J,H,Ts,sweeps)


# In[ ]:


# Plot time evolution
N_indices = range(len(Ns))
T_indices = [0,19,39]
startpt = 0
stoppt = sweeps
plot_time_series(magnetisation_per_site,Ns,N_indices,Ts,T_indices,startpt,stoppt,'Sweeps','Magnetisation per site')


# In[ ]:


# Plot run time vs N to investigate program complexity
run_time_data = np.zeros(len(Ns))
for N_index in N_indices:
    run_time_data[N_index] = np.average(run_time[N_index,:])
plt.plot(Ns,run_time_data,'-o')
plt.xlabel('Lattice size N')
plt.ylabel('Run time (s)')
plt.savefig('plots/Run time vs Lattice size.pdf')


# In[ ]:


# How total magnetisation fluctuates with time when system is in equilibrium
magnetisation_autocovariance = calculate_autocovariation(magnetisation,Ns,Ts,sweeps,equilibrating_sweeps)


# In[ ]:


# Plot autocovariance of total magnetisation
N_indices = range(len(Ns))
T_indices = range(26,31)
startpt = 0
stoppt = sweeps
plot_time_series(magnetisation_autocovariance,Ns,N_indices,Ts,T_indices,startpt,stoppt,'tau','Autocorrelation')


# In[ ]:


# Calculate thermodynamic variables
average_magnetisation = calculate_thermodynamic_variable(np.abs(magnetisation),Ns,Ts,equilibrating_sweeps)
average_energy = calculate_thermodynamic_variable(energy,Ns,Ts,equilibrating_sweeps)
average_susceptibility = calculate_derivative_thermodynamic_average(np.abs(magnetisation),Ns,Ts,1,equilibrating_sweeps)
average_heat_capacity = calculate_derivative_thermodynamic_average(energy,Ns,Ts,2,equilibrating_sweeps)


# In[ ]:


# Plot thermodynamic variables
N_indices = range(len(Ns))
T_indices = range(len(Ts))
plot_temperature_dependence(average_magnetisation,Ns,N_indices,Ts,T_indices,'Temperature','Average Magnetisation')
plot_temperature_dependence(average_energy,Ns,N_indices,Ts,T_indices,'Temperature','Average Energy')
plot_temperature_dependence(average_susceptibility,Ns,N_indices,Ts,T_indices,'Temperature','Average Susceptibility')
plot_temperature_dependence(average_heat_capacity,Ns,N_indices,Ts,T_indices,'Temperature','Average Heat Capacity')


# In[ ]:


# Fit magnetisation
N_indices = range(len(Ns))
T_indices = range(len(Ts))
parameters, errors = data_fitting(average_magnetisation,Ns,N_indices,Ts,T_indices,[0.,2.269,0.125])
print(parameters, errors)


# In[ ]:


# Plot fitted vs measured data
for N_index in N_indices:
    plt.plot(Ts[:],average_magnetisation[N_index,:],'-o',label='measured data')
    plt.plot(Ts[T_indices],shape_function(Ts[T_indices],parameters[N_index,0],parameters[N_index,1],parameters[N_index,2]),label='fitted data, N = {0}, Tc = {1:.3f}, beta = {2:.3f}'.format(Ns[N_index],parameters[N_index,1],parameters[N_index,2]))
    plt.legend(loc='best')
    plt.xlabel('Temperature')
    plt.ylabel('Average Magnetisation')
    plt.savefig('plots/Fitted vs Measured Magnetisation N = {0}.pdf'.format(Ns[N_index]))
    plt.figure()

