from random import uniform, randint
import random
from math import exp
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import Counter
import turtle

# Global variables
J = 1 # Exchange energy
mu = 1 # Magnetic moment
number_of_neighbours = 4
random_seed = 13 # Fix random seed in spin initialization for reproducability
hot_start = False # Initialize with hot start or not
nsweeps = 8000 # Number of sweeps

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

def get_energy_difference(index_to_flip, s, neighbours_dictionary, H):
    """
    :param H: (float) Applied magnetic field
    :param index_to_flip: (int) the site index to consider flipping its spin
    :param s: (list) spin list of the lattice
    :param neighbours_dictionary: (dict) holds indexes for each site's neighbours
    :return: (float) Total energy change of changing that site's spin
    """
    sum_of_neighbours = 0
    for neighbour_index in neighbours_dictionary[index_to_flip]:
        sum_of_neighbours += s[neighbour_index]
    total_change = 2*s[index_to_flip]*sum_of_neighbours + 2*s[index_to_flip]*mu*H
    return total_change

def metropolis(N, s, neighbours_dictionary, H, b):
    """
    The metropolis algorithm as a markov chain monte carlo simulation algorithm that modifies the spin state of the
    lattice and gives a new state by choosing N (= L x L) sites at random and checking through the energy if it will
    flip the site's spin or not. The dE_4 and dE_8 are the 2 cases when the spin will be flipped and the numbers 4, 8
    represent the corresponding change in energy so that we don't calculate many times an exponential term
    :param b: (float) 1/Temperature
    :param H: (float) Applied magnetic field
    :param N: (int) total number of sites
    :param s: (list) spin list
    :param neighbours_dictionary: (dict) holds indexes for neighbours
    :return: (void) does not return anything but changes the state of s
    """
    for i in range(N):
        site_index = randint(0, N-1)
        dE = get_energy_difference(site_index, s, neighbours_dictionary, H)
        val = exp(-dE*b)
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

def get_energy(N, s, neighbours_dictionary, H):
    """
    :param H: (float) Applied magnetic field
    :param N: (int) total number of sites
    :param s: (list) spin list
    :param neighbours_dictionary: (dict) holds indexes for neighbours
    :return: (float) Total energy through the Hamiltonian
    """
    sum1 = 0
    sum2 = 0
    for i in range(N):
        for j in range(number_of_neighbours):
            sum1 += s[i]*s[neighbours_dictionary[i][j]]
        if H != 0:
            sum2 = get_magnetisation(N, s)
    total_energy = (-J*sum1 - mu*H*sum2)/2
    return total_energy

def get_average_energy(N, s, neighbours_dictionary, H):
    return get_energy(N, s, neighbours_dictionary, H)/(2*N)

def get_average_magnetisation(N, s):
    """
    :param N: (int) total number of sites
    :param s: (list) spin list
    :return: (float) Magnetisation per site
    """
    return get_magnetisation(N, s)/N

def spin_configuration_2D(s, L):

    x, y = 0, L-1
    s_2D = [[0 for i in range(L)] for j in range(L)]
    for idx in range(len(s)):

        if (idx + 1) % L == 0:
            s_2D[x][y] = s[idx]
            x = 0
            y -= 1
        else:
            s_2D[x][y] = s[idx]
            x += 1
    return s_2D

def draw_lattice(s, L, T, avg_clust_size):

    s_2D = []

    for i in range(L):
        s_2D.append(s[i*L:(i+1)*L])

    s_2D_array = np.array(s_2D)
    plt.imshow(s_2D_array, cmap = 'binary')
    plt.title(f'L = {L}, T = {T:.2f}, Average cluster size = {avg_clust_size:.2f}')
    plt.show()

def simulation():
    """
    The simulation function gathers data for average magnetisation vs temperature for different L dimensions of the
    lattice. It stores the data in a pandas dataframe and makes a plot of all L in the same average magnetisation vs
    temperature graph
    :return: (void)
    """
    # Temperature values in Kelvin
    T_values = [2.4, 2.6, 3]
    # Dimension values
    L_values = [10]
    # Applied magnetic field values
    extra_H_neg = list(np.linspace(-0.05128, -0.05385, 5))
    H_val = list(np.linspace(0, 1.0, 21))
    H_val.extend(H_val[::-1])
    H_neg = list(np.linspace(0, -1.0, 21))
    H_neg.extend(extra_H_neg)
    H_neg.sort(reverse=True)
    H_neg.extend(H_neg[::-1])
    H_val.extend(H_neg)
    H_values = [round(H_val[i], 10) for i in range(len(H_val))]
    tmp_list = [H_values[0]]

    for i in range(len(H_values) - 1):

        if H_values[i] != H_values[i + 1]:
            tmp_list.append(H_values[i + 1])

    H_values = tmp_list
    # Independent sampling
    thermalisation_sweeps = 3000
    sample_every = 20

    with open('hysteresis.txt', 'w') as file:

        for L in L_values:

            for T in T_values:

                N = L ** 2
                s = get_spin_list(hot_start, N)
                neighbours_dictionary = get_neighbours_index(N)
                b = 1/T  # Constant: 1 / temperature

                for H in H_values:

                    start = time.process_time()
                    average_magnetisation_list = []
                    energy_per_site_list = []

                    for i in range(nsweeps):

                        metropolis(N, s, neighbours_dictionary, H, b)
                        if i < thermalisation_sweeps:
                            continue
                        elif i % sample_every == 0:
                            mag_per_site = get_average_magnetisation(N, s)
                            average_magnetisation_list.append(mag_per_site)
                            energy_per_site_list.append(get_average_energy(N, s, neighbours_dictionary, H))

                    mean_average_magnetisation = np.average(average_magnetisation_list)
                    mean_energy = np.average(energy_per_site_list)
                    file.write(f'{L},{T},{H},{mean_average_magnetisation},{mean_energy}\n')
                    time_for_sample = time.process_time() - start
                    print(f'L = {L}, T = {T}, H = {H}, <m> = {mean_average_magnetisation:.5f}, <E> = {mean_energy:.2f} --> Time for sample = {time_for_sample:.2f} seconds')

def hysteresis_plotter():

    magnetisation_per_site_list = []
    energy_per_site = []
    L_list = []
    T_list = []
    H_list = []

    with open('hysteresis.txt', 'r') as file:

        line = file.readline()
        count = 0

        while line:
            row = line.strip().split(',')
            L_list.append(int(row[0]))
            T_list.append(float(row[1]))
            magnetisation_per_site_list.append(float(row[3]))
            energy_per_site.append(float(row[4]))
            if count == 3:
                line = file.readline()
                continue
            else:
                H_list.append(float(row[2]))
                if float(row[2]) == 0:
                    count += 1
            line = file.readline()

        L_values = list(Counter(L_list).keys())
        T_values = list(Counter(T_list).keys())
        H_values = H_list
        H_len = len(H_values)
        T_str_values = ['T = ' + str(T_values[i]) for i in range(len(T_values))]

        for i in range(len(T_values)):
            plt.scatter(H_values, magnetisation_per_site_list[i*H_len:(i+1)*H_len], marker = '+', s = None)
        plt.title(f'Hysteresis loop (L = {L_values[0]})\nMagnetisation per site vs applied field')
        plt.xlabel('H')
        plt.ylabel('<m>')
        plt.legend(T_str_values)
        plt.grid()
        # plt.savefig('hysteresis_mag.pdf', bbox = 'tight')
        plt.show()

        for i in range(len(T_values)):
            plt.scatter(H_values, energy_per_site[i * H_len:(i + 1) * H_len], marker = '+', s = None)
        plt.title(f'Energy per site vs applied field (L = {L_values[0]})')
        plt.xlabel('H')
        plt.ylabel('<E>')
        plt.grid()
        plt.legend(T_str_values)
        # plt.savefig('hysteresis_energy.pdf', bbox = 'tight')
        plt.show()

#simulation()
hysteresis_plotter()
