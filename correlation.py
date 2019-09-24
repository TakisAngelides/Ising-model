from random import uniform, randint
import random
from math import exp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Global variables
L = 30 # Lattice dimension
N = L**2 # Total number of nodes in lattice
J = 1 # Exchange energy
H = 0 # Applied magnetic field strength
mu = 1 # Magnetic moment
M = 0 # Total magnetisation
T = 1.0 # Temperature in Kelvin
b = 1/T # Constant: 1 / temperature
dE_4 = exp(-4*b) # Probability for energy change of +4
dE_8 = exp(-8*b) # Probability for energy change of +8
number_of_neighbours = 4
random_seed = 13 # Fix random seed in spin initialization for reproducability
hot_start = True # Initialize with hot start
nsweeps = 30000 # Number of sweeps

def get_spin_list(is_hot):
    """
    :param is_hot: (bool) If is_hot is true make hot start otherwise make cold start
    :return: (list) containing +/- 1 representing the spins on the lattice
    """
    random.seed(random_seed)
    # A hot start means start with +/- 1 with a 50-50 chance on each site
    if is_hot:
        s_local = []
        for i in range(N):
            rdm_num = uniform(0, 1)
            if rdm_num > 0.5:
                s_local.append(1)
            else:
                s_local.append(-1)
    # A cold start means start with 1s on all sites
    else:
        s_local = [1] * N
    return s_local

def get_neighbours_index():
    """
    :return: (dict) containing as keys the index of the site on the lattice and as values a list containing the indexes
    of its neighbours
    """
    neighbours_dict = {}
    for i in range(N):
        # store index of neighbours in the values for each node (key, i) in the lattice
        # in the form left, right, top, bottom with periodic boundary conditions
        if i % L == 0:
            left = i + L - 1
        else:
            left = i - 1
        if (i + 1) % L == 0:
            right = i - L + 1
        else:
            right = i + 1
        if i - L < 0:
            top = i - L + N
        else:
            top = i - L
        if i + L >= N:
            bottom = i + L - N
        else:
            bottom = i + L

        neighbours_dict[i] = [left, right, top, bottom]

    return neighbours_dict

s = get_spin_list(hot_start)
neighbours_dictionary = get_neighbours_index()

def get_energy_difference(index_to_flip):
    """
    :param index_to_flip: (int) the site index to consider flipping its spin
    :return: (float) Total energy change of changing that site's spin
    """
    sum_of_neighbours = 0
    for neighbour_index in neighbours_dictionary[index_to_flip]:
        sum_of_neighbours += s[neighbour_index]
    total_change = 2*s[index_to_flip]*sum_of_neighbours # Works out from the Hamiltonian of the before and after states
    return total_change

def metropolis():
    """
    The metropolis algorithm as a markov chain monte carlo simulation algorithm that modifies the spin state of the
    lattice and gives a new state by choosing N (= L x L) sites at random and checking through the energy if it will
    flip the site's spin or not. The dE_4 and dE_8 are the 2 cases when the spin will be flipped and the numbers 4, 8
    represent the corresponding change in energy so that we don't calculate many times an exponential term
    :return: (void) does not return anything but changes the state of s
    """
    for i in range(N):
        site_index = randint(0,N-1)
        dE = get_energy_difference(site_index)
        rdm_num = uniform(0, 1)
        if dE <= 0:
            s[site_index] *= -1
        elif dE == 4:
            if dE_4 > rdm_num:
                s[site_index] *= -1
            else:
                continue
        elif dE == 8:
            if dE_8 > rdm_num:
                s[site_index] *= -1
            else:
                continue

def get_magnetisation():
    """
    :return: (float) Total magnetisation
    """
    magnetisation_total = 0
    for i in range(N):
        magnetisation_total += s[i]
    return magnetisation_total

def get_energy():
    """
    :return: (float) Total energy through the Hamiltonian
    """
    sum1 = 0
    sum2 = 0
    for i in range(N):
        for j in range(number_of_neighbours):
            sum1 += s[i]*s[neighbours_dictionary[i][j]]
        if H != 0:
            sum2 = get_magnetisation()
    total_energy = (-J*sum1 - mu*H*sum2)/2
    return total_energy

def get_average_magnetisation():
    """
    :return: (float) Magnetisation per site
    """
    return abs(get_magnetisation())/N

def get_autocovariance(M_list, tau):
    """
    :param M_list: (list) holding average magnetisation for each sweep neglected pre-thermalised samples
    :param tau: (int) time lag which is the input to the autocovariance formula
    :return: (float) autocovariance for the time lag tau
    """
    mean = np.average(M_list)
    if len(M_list)-tau<=0:
        print(f'The length of M_list in auto_covariance is {len(M_list)} and of tau {tau}')
    autocovariance_list = [(M_list[t] - mean)*(M_list[t+tau] - mean) for t in range(len(M_list) - tau)]
    return  np.average(autocovariance_list)

def get_autocorrelation(M_list, tau):
    """
    :param M_list: (list) holding average magnetisation for each sweep neglected pre-thermalised samples
    :param tau: (int) time lag which is the input to the autocovariance formula
    :return: (float) autocorrelation for the time lag tau
    """
    A_0 = get_autocovariance(M_list, 0)
    A_tau = get_autocovariance(M_list, tau)
    return  A_tau/A_0

def get_target_value_index(autocorrelation_list, target_value):
    """
    When the autocorrelation falls below 1/e return the index
    :param autocorrelation_list: (list) autocorrelation for different tau values
    :param target_value: (float) 1/e of the initial autocorrelation value
    :return: (int) Index of tau that gives 1/e of initial autocorrelation
    """
    target_index = -1
    for i in range(len(autocorrelation_list)):
        if autocorrelation_list[i] < target_value:
            target_index = i
            break
        else:
            continue
    return target_index

def simulation():
    """
    Evolves the systems with nsweeps and records average magnetisation for every state. The the simulation
    finds the tau that makes autocorrelation fall to 1/e called the constant of decorrelation and plots
    autocorrelation vs tau
    :return: (void)
    """
    avg_magnetisation_list = []
    autocorrelation_list = []
    thermalisation_sweeps = 5000
    sample_every = 50

    for t in range(nsweeps):
        metropolis()
        if t < thermalisation_sweeps:
            continue
        else:
            if t % sample_every == 0:
                avg_magnetisation_list.append(get_average_magnetisation())

    tau_list = np.arange(0, 50, 1)

    for tau in tau_list:
        autocorrelation_list.append(get_autocorrelation(avg_magnetisation_list, tau))

    target_value = 1/np.e
    index = get_target_value_index(autocorrelation_list, target_value)
    target_tau = tau_list[index]

    plt.plot(tau_list, autocorrelation_list, color = 'k')
    plt.axhline(y = target_value, linestyle = '--', color = 'r')
    plt.axvline(x = target_tau, linestyle = '--', color='b')
    plt.title(f'Autocorrelation \u03b1(\u03C4) vs \u03C4\nTemperature: {T}K, L = {L}')
    plt.xlabel('\u03C4')
    plt.ylabel('Autocorrelation \u03b1(\u03C4)')
    plt.legend(['\u03b1(\u03C4)', '1/e', f'\u03C4 at 1/e = {target_tau}'])
    plt.grid(True)
    # plt.savefig(f'autocorrelation_vs_tau_{T}.pdf', bbox = 'tight')
    plt.show()

simulation()