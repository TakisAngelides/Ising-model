from random import uniform, randint
import random
from math import exp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Global variables
L = 10 # Lattice dimension
N = L**2 # Total number of nodes in lattice
J = 1 # Exchange energy
H = -2.5 # Applied magnetic field strength
mu = 1 # Magnetic moment
M = 0 # Total magnetisation
T = 0.9 # Temperature in Kelvin
b = 1/T # Constant: 1 / temperature
dE_4 = exp(-4*b) # Probability for energy change of +4
dE_8 = exp(-8*b) # Probability for energy change of +8
number_of_neighbours = 4
random_seed = 31 # Fix random seed in spin initialization for reproducability
hot_start = True # Initialize with hot start
nsweeps = 2000 # Number of sweeps

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
            rdm_num = uniform(0, 1) # Get a float between 0 and 1 from a uniform distribution
            if rdm_num > 0.5:
                s_local.append(1)
            else:
                s_local.append(-1)
    # A cold start means start with 1s on all sites
    else:
        s_local = [1] * N # Creates a list of N (+1)'s
    return s_local

def get_neighbours_index():
    """
    :return: (dict) containing as keys the index of the site on the lattice and as values a list containing the indexes
    of its neighbours
    """
    neighbours_dict = {} # Index of site -> indexes of neighbours of that site (key -> value)
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

s = get_spin_list(hot_start) # Initialise a lattice
neighbours_dictionary = get_neighbours_index() # Initialise the dictionary holding the neighbours

def get_energy_difference(index_to_flip):
    """
    :param index_to_flip: (int) the site index to consider flipping its spin
    :return: (float) Total energy change of changing that site's spin
    """
    sum_of_neighbours = 0
    for neighbour_index in neighbours_dictionary[index_to_flip]:
        sum_of_neighbours += s[neighbour_index]
    total_change = 2*s[index_to_flip]*sum_of_neighbours + 2*s[index_to_flip]*mu*H
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
        site_index = randint(0, N-1)
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

def get_average_energy():
    """
    :return: (float) Average energy through the Hamiltonian
    """
    sum1 = 0
    sum2 = 0
    for i in range(N):
        for j in range(number_of_neighbours):
            sum1 += s[i]*s[neighbours_dictionary[i][j]]
        if H != 0:
            sum2 = get_magnetisation()
    total_energy = (-J*sum1 - mu*H*sum2)/2
    return total_energy/(2*N)

def get_average_magnetisation():
    """
    :return: (float) Magnetisation per site
    """
    return get_magnetisation()/N

def simulation():
    """
    Goes through all sweeps modifying the state with the metropolis function and logging total
    magnetisation and energy. Then calculates the mean and standard deviation of the total magnetisation after
    thermalisation to quantify its fluctuations. The it plots time vs total and average energy and time vs magnetisation
    :return: (void)
    """
    total_magnetisation = []
    total_energy = []
    nsweeps_values = np.arange(0, nsweeps)

    # Evolve the system and save magnetisation and energy for each new state
    for i in range(nsweeps):
        metropolis()
        total_magnetisation.append(get_magnetisation())
        total_energy.append(get_energy())

    # Quantify fluctuations after thermalisation of 1000 sweeps
    mean = np.average(total_magnetisation[1000:])
    standard_deviation = np.std(total_magnetisation[1000:])

    plt.plot(nsweeps_values, total_magnetisation, color = 'k')
    plt.xlabel('Time (sweeps)')
    plt.ylabel('Total magnetisation')
    plt.title(f'Time vs total magnetisation\nThermalised mean: {mean:.3f}, Std: {standard_deviation:.3f}\nTemperature: {T}K, L = {L}')
    plt.grid(True)
    # plt.savefig(f'time_vs_total_mag_{T}.pdf', bbox = 'tight')
    plt.show()

    plt.plot(nsweeps_values, np.array(total_magnetisation)/N, color='k')
    plt.xlabel('Time (sweeps)')
    plt.ylabel('Average magnetisation')
    plt.title(
        f'Time vs average magnetisation\nTemperature: {T}K, L = {L}')
    plt.grid(True)
    # plt.savefig(f't_vs_avg_mag_{T}.pdf', bbox = 'tight')
    plt.show()

    plt.plot(nsweeps_values, total_energy, color = 'k')
    plt.xlabel('Time (sweeps)')
    plt.ylabel('Total energy')
    plt.title(f'Time vs total energy\nTemperature: {T}K, L = {L}')
    plt.grid(True)
    # plt.savefig(f'time_vs_total_energy_{T}.pdf', bbox = 'tight')
    plt.show()

    average_energy = np.array(total_energy)/(2*N)
    plt.plot(nsweeps_values, average_energy, color='k')
    plt.xlabel('Time (sweeps)')
    plt.ylabel('Average energy')
    plt.title(f'Time vs average energy\nTemperature: {T}K, L = {L}')
    plt.grid(True)
    # plt.savefig(f'time_vs_avg_energy_{T}.pdf', bbox = 'tight')
    plt.show()

simulation()