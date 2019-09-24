from random import uniform, randint
import random
from math import exp
from lattice_3d_plotter import lattice_3d_plotter
import numpy as np
import time

# Global variables
H = 0 # Applied magnetic field strength
mu = 1 # Magnetic moment
M = 0 # Total magnetisation
J = 1 # Exchange energy
number_of_neighbours = 6 # Number of neighbours in 3D
random_seed = 10 # Fix random seed in spin initialization for reproducability
hot_start = False # Initialize with hot start or not
nsweeps = 4000 # Number of sweeps

def get_spin_list(is_hot, N):
    """
    :param is_hot: (bool) If is_hot is true make hot start otherwise make cold start
    :param N: (int) N = L x L, total number of sites
    :return: (list) containing +/- 1 representing the spins on the lattice
    """
    random.seed(random_seed)
    L = int(N**(1/2))
    # A hot start means start with +/- 1 with a 50-50 chance on each site
    if is_hot:
        s_local = []
        for i in range(L**3):
            rdm_num = uniform(0, 1)
            if rdm_num > 0.5:
                s_local.append(1)
            else:
                s_local.append(-1)
    # A cold start means start with 1s on all sites
    else:
        s_local = [1] * (L**3)
    return s_local

def get_neighbours_index(N):
    """
    :param N: N = L x L
    :return: (dictionary) key is site value is list of neighbour indices
    """
    L = int(N**(1/2))
    neighbours_dictionary = {}
    lattice_indices = [[0 for k in range(N)] for j in range(L)]

    for i in range(L):
        for j in range(N):
            lattice_indices[i][j] = j + i*N
    for i in range(L):
        for j in range(N):

            idx = j + i*N

            if idx % L == 0:
                left = lattice_indices[i][j + L - 1]
            else:
                left = lattice_indices[i][j - 1]
            if (idx + 1) % L == 0:
                right = lattice_indices[i][j - L + 1]
            else:
                right = lattice_indices[i][j + 1]
            if idx - L < 0:
                top = lattice_indices[i][j - L + N]
            else:
                top = lattice_indices[i][j - L]
            if idx + L >= N:
                bottom = lattice_indices[i][j + L - N]
            else:
                bottom = lattice_indices[i][j + L]
            if i == 0:
                up = lattice_indices[L-1][j]
            else:
                up = lattice_indices[i-1][j]
            if i == L-1:
                down = lattice_indices[0][j]
            else:
                down = lattice_indices[i+1][j]

            neighbours_dictionary[idx] = [left, right, top, bottom, up, down]
    return neighbours_dictionary

def get_energy_difference(index_to_flip, s, neighbours_dictionary):
    """
    :param index_to_flip: (int) the site index to consider flipping its spin
    :param s: (list) spin list of the lattice
    :param neighbours_dictionary: (dict) holds indexes for each site's neighbours
    :return: (float) Total energy change of changing that site's spin
    """
    sum_of_neighbours = 0
    for neighbour_index in neighbours_dictionary[index_to_flip]:
        sum_of_neighbours += s[neighbour_index]
    total_change = 2*s[index_to_flip]*sum_of_neighbours
    return total_change

def metropolis(N, s, neighbours_dictionary, b):
    """
    :param N: (int) total number of sites
    :param s: (list) spin list
    :param neighbours_dictionary: (dict) holds indexes for neighbours
    :param b: (float) 1/Temperature
    :return: (void) does not return anything but changes the state of s
    """
    L = int(N**(1/2))
    for i in range(L**3):
        site_index = randint(0, (L**3)-1)
        dE = get_energy_difference(site_index, s, neighbours_dictionary)
        val = exp(-dE * b)
        rdm_num = uniform(0, 1)
        if dE <= 0:
            s[site_index] *= -1
        elif val > rdm_num:
            s[site_index] *= -1
        else:
            continue

def get_magnetisation(N, s):
    """
    :param N: (int) total number of sites
    :param s: (list) spin list
    :return: (float) Total magnetisation
    """
    magnetisation_total = 0
    L = int(N**(1/2))
    for i in range(L**3):
        magnetisation_total += s[i]
    return magnetisation_total

def get_average_magnetisation(N, s):
    """
    :param N: (int) total number of sites
    :param s: (list) spin list
    :return: (float) Magnetisation per site
    """
    L = int(N**(1/2))
    return abs(get_magnetisation(N, s))/(L**3)

def get_energy(N, s, neighbours_dictionary):
    """
    :param N: (int) total number of sites
    :param s: (list) spin list
    :param neighbours_dictionary: (dict) holds indexes for neighbours
    :return: (float) Total energy through the Hamiltonian
    """
    sum1 = 0
    sum2 = 0
    L = int(N**(1/2))
    for i in range(L**3):
        for j in range(number_of_neighbours):
            sum1 += s[i]*s[neighbours_dictionary[i][j]]
        if H != 0:
            sum2 = get_magnetisation(N, s)
    total_energy = (-J*sum1 - mu*H*sum2)/2
    return total_energy

def get_average_energy(N, s, neighbours_dictionary):
    L = int(N**(1/2))
    return get_energy(N, s, neighbours_dictionary)/(3*(L**3))

def simulation():

    with open('lattice_3d.txt', 'w') as file:

        # Temperature values in Kelvin
        T_values = np.linspace(2, 6, 51)
        # Dimension values
        L_values = [12, 14]
        # Sampling
        thermalisation_sweeps = 2000
        sample_every = 50

        for L in L_values:

            N = L**2
            s = get_spin_list(hot_start, N)
            neighbours_dictionary = get_neighbours_index(N)

            for T in T_values:

                start = time.process_time()
                b = 1 / T  # Constant: 1 / temperature
                tmp_magnetisation = []
                tmp_energy = []

                for i in range(nsweeps):
                    metropolis(N, s, neighbours_dictionary, b)
                    if i < thermalisation_sweeps:
                        continue
                    elif i % sample_every == 0:
                        tmp_magnetisation.append(get_average_magnetisation(N, s))
                        tmp_energy.append(get_average_energy(N, s, neighbours_dictionary))

                time_for_sample = time.process_time() - start
                file.write(f'{L},{T},{np.average(tmp_magnetisation)},{np.average(tmp_energy)}\n')
                print(f'L = {L}, T = {T:.2f}, <m> = {np.average(tmp_magnetisation):.5f}, <E> = {np.average(tmp_energy):.5f}, Time for sample = {time_for_sample:.2f} seconds')

simulation()
lattice_3d_plotter()