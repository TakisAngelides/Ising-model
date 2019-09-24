from random import uniform, randint
import random
from math import exp
from nearest_neighbours_plottter import nearest_neighbours_plotter
import numpy as np
import time

# Global variables
H = 0 # Applied magnetic field strength
mu = 1 # Magnetic moment
M = 0 # Total magnetisation
random_seed = 10 # Fix random seed in spin initialization for reproducability
hot_start = False # Initialize with hot start or not
nsweeps = 2000 # Number of sweeps

def get_spin_list(is_hot, N):
    """
    :param is_hot: (bool) If is_hot is true make hot start otherwise make cold start
    :param N: (int) N = L x L, total number of sites
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

def get_neighbours_index(N):
    """
    :param N: (int) N = L x L, total number of sites
    :return: (dict) containing as keys the index of the site on the lattice and as values a list containing the indexes
    of its neighbours
    """
    neighbours_dict = {}
    L = int(N**(1/2))
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

def get_nearest_neighbours_index(N):
    """
    :param N: (int) Total number of sites
    :return: (dictionary) Keys - sites indices, values - next nearest neighbours' indices
    """
    L = int(N ** (1 / 2)) # Lattice dimension
    nearest_neighbours_dictionary = {} # Dictionary to return

    for i in range(N):
        # Up-left neighbour
        if i == 0: # If the site is index 0
            upleft = N - 1
        elif i % L == 0: # If the site is in the first column
            upleft = i - 1
        elif i - L < 0: # If the site is in the first row
            upleft = i - L + N - 1
        else:
            upleft = i - L - 1
        # Up-right neighbour
        if i == L - 1: # If the site is at the top right corner
            upright = L * (L - 1)
        elif (i + 1) % L == 0: # If the site is in the last column
            upright = i - 2 * L + 1
        elif i - L < 0: # If the site is in the first row
            upright = i - L + N + 1
        else:
            upright = i - L + 1
        # Down-left neighbour
        if i == L * (L - 1): # If the site is at the bottom left corner
            downleft = L - 1
        elif i + L >= N: # If the site is in the last row
            downleft = i + L - N - 1
        elif i % L == 0: # If the site is in the first column
            downleft = i + 2 * L - 1
        else:
            downleft = i + L - 1
        # Down-right neighbour
        if i == N - 1: # If the site is at the down right corner
            downright = 0
        elif (i + 1) % L == 0: # If the site is in the last column
            downright = i + 1
        elif i + L >= N: # If the site is in the last row
            downright = i + L - N + 1
        else:
            downright = i + L + 1

        nearest_neighbours_dictionary[i] = [upleft, upright, downleft, downright]

    return nearest_neighbours_dictionary

def get_energy_difference(index_to_flip, s, neighbours_dictionary, nearest_neighbours_dictionary, J):
    """
    :param index_to_flip: (int) the site index to consider flipping its spin
    :param s: (list) spin list of the lattice
    :param neighbours_dictionary: (dict) holds indexes for each site's neighbours
    :param nearest_neighbours_dictionary: (dict) holds indexes for each site's nearest neighbours
    :param J: (float) exchange energy
    :return: (float) Total energy change of changing that site's spin
    """
    sum_of_neighbours = 0
    sum_of_nearest_neighbours = 0
    for neighbour_index in neighbours_dictionary[index_to_flip]:
        sum_of_neighbours += s[neighbour_index]
    for nearest_neighbour_index in nearest_neighbours_dictionary[index_to_flip]:
        sum_of_nearest_neighbours += s[nearest_neighbour_index]
    total_change = 2*s[index_to_flip]*sum_of_neighbours + 2*J*s[index_to_flip]*sum_of_nearest_neighbours
    return total_change

def metropolis(N, s, neighbours_dictionary, nearest_neighbours_dictionary, J, b):
    """
    :param N: (int) total number of sites
    :param s: (list) spin list
    :param neighbours_dictionary: (dict) holds indexes for neighbours
    :param nearest_neighbours_dictionary: (dict) holds indexes for nearest neighbours
    :param J: (float) exchange energy for next-nearest neighbours only (J=1 for neighbours)
    :param b: (float) 1/Temperature
    :return: (void) does not return anything but changes the state of s
    """
    for i in range(N):
        site_index = randint(0, N-1)
        dE = get_energy_difference(site_index, s, neighbours_dictionary, nearest_neighbours_dictionary, J)
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
    for i in range(N):
        magnetisation_total += s[i]
    return magnetisation_total

def get_average_magnetisation(N, s):
    """
    :param N: (int) total number of sites
    :param s: (list) spin list
    :return: (float) Magnetisation per site
    """
    return abs(get_magnetisation(N, s))/N

def simulation():

    with open('nearest_neighbours1.txt', 'w') as file:

        # Temperature values in Kelvin
        T_values = np.linspace(1, 10, 51)
        # Dimension values
        L_values = [64]
        # Exchange energy values
        J_values = [1, 0.7, 0.5]
        # Sampling
        thermalisation_sweeps = 1000
        sample_every = 50

        for L in L_values:

            N = L**2
            s = get_spin_list(hot_start, N)
            neighbours_dictionary = get_neighbours_index(N)
            nearest_neighbours_dictionary = get_nearest_neighbours_index(N)

            for J in J_values:

                for T in T_values:

                    start = time.process_time()
                    b = 1 / T  # Constant: 1 / temperature
                    tmp_magnetisation = []

                    for i in range(nsweeps):
                        metropolis(N, s, neighbours_dictionary, nearest_neighbours_dictionary, J, b)
                        if i < thermalisation_sweeps:
                            continue
                        elif i % sample_every == 0:
                            tmp_magnetisation.append(get_average_magnetisation(N, s))

                    time_for_sample = time.process_time() - start
                    file.write(f'{L},{J},{T},{np.average(tmp_magnetisation)}\n')
                    print(f'L = {L}, J = {J}, T = {T:.2f}, <m> = {np.average(tmp_magnetisation):.5f}, Time for sample = {time_for_sample:.2f} seconds')

simulation()
nearest_neighbours_plotter()