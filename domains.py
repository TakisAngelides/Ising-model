from random import uniform, randint
import random
from math import exp
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import time
from collections import Counter
from queue import Queue

# Global variables
J = 1 # Exchange energy
H = 0
mu = 1 # Magnetic moment
number_of_neighbours = 4
random_seed = 13 # Fix random seed in spin initialization for reproducability
hot_start = True # Initialize with hot start or not
nsweeps = 10000 # Number of sweeps

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
    total_change = 2*s[index_to_flip]*sum_of_neighbours + 2*s[index_to_flip]*mu*H
    return total_change

def metropolis(N, s, neighbours_dictionary, b):
    """
    The metropolis algorithm as a markov chain monte carlo simulation algorithm that modifies the spin state of the
    lattice and gives a new state by choosing N (= L x L) sites at random and checking through the energy if it will
    flip the site's spin or not. The dE_4 and dE_8 are the 2 cases when the spin will be flipped and the numbers 4, 8
    represent the corresponding change in energy so that we don't calculate many times an exponential term
    :param b: (float) 1/Temperature
    :param N: (int) total number of sites
    :param s: (list) spin list
    :param neighbours_dictionary: (dict) holds indexes for neighbours
    :return: (void) does not return anything but changes the state of s
    """
    for i in range(N):
        site_index = randint(0, N-1)
        dE = get_energy_difference(site_index, s, neighbours_dictionary)
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

def get_energy(N, s, neighbours_dictionary):
    """
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

def get_average_energy(N, s, neighbours_dictionary):
    return get_energy(N, s, neighbours_dictionary)/(2*N)

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

def bfs(s, index_to_start, neighbours_dictionary):
    """
    :param s: (list) lattice spin list
    :param index_to_start: (int) index to start calculating a cluster from
    :param neighbours_dictionary: (dictionary) holds indices for the site's neighbours
    :return: (int, list) cluster size, indices of sites that are now saved in a calculated cluster
    """
    cluster_type = s[index_to_start]
    seen = [False] * len(s)
    to_explore = Queue()
    to_explore.put(index_to_start)
    seen[index_to_start] = True
    cluster_size = 1
    cluster_indices = [index_to_start]

    while not to_explore.empty():
        idx = to_explore.get()
        for neighbour_index in neighbours_dictionary[idx]:
            if not seen[neighbour_index] and s[neighbour_index] == cluster_type:
                to_explore.put(neighbour_index)
                seen[neighbour_index] = True
                cluster_size += 1
                cluster_indices.append(neighbour_index)

    return cluster_size, cluster_indices

def get_average_cluster_size(s, neighbours_dictionary):

    cluster_size, cluster_indices = bfs(s, 0, neighbours_dictionary)

    cluster_size_list = [cluster_size]

    for index_to_start in range(len(s)):
        index_to_start += 1
        if index_to_start > len(s)-1:
            break
        elif index_to_start in cluster_indices:
            continue
        else:
            cluster_size, cluster_indices_tmp = bfs(s, index_to_start, neighbours_dictionary)
            cluster_size_list.append(cluster_size)
            cluster_indices.extend(cluster_indices_tmp)

    final_cluster_size_list = []
    maximum_value = max(cluster_size_list)

    for i in range(len(cluster_size_list)):

        if cluster_size_list[i] > maximum_value/2:
            final_cluster_size_list.append(cluster_size_list[i])

    return np.average(final_cluster_size_list)

def simulation():
    """
    The simulation function gathers data for average magnetisation vs temperature for different L dimensions of the
    lattice. It stores the data in a pandas dataframe and makes a plot of all L in the same average magnetisation vs
    temperature graph
    :return: (void)
    """
    # Temperature values in Kelvin
    T_extra = list(np.linspace(2.2, 2.5, 11))
    T_values = list(np.linspace(2, 2.8, 31))
    T_values.extend(T_extra)
    T_values.sort()
    # Dimension values
    L_values = [12, 24, 32]
    # Independent sampling
    thermalisation_sweeps = 5000
    sample_every = 50

    with open('domains.txt', 'w') as file:

        for L in L_values:

            N = L**2
            s = get_spin_list(hot_start, N)
            neighbours_dictionary = get_neighbours_index(N)

            for T in T_values:

                b = 1/T  # Constant: 1 / temperature
                cluster_size_list = []
                start = time.process_time()

                for i in range(nsweeps):
                    metropolis(N, s, neighbours_dictionary, b)
                    if i < thermalisation_sweeps:
                        continue
                    elif i % sample_every == 0:
                        cluster_size_list.append(get_average_cluster_size(s, neighbours_dictionary))

                cluster_std = np.std(cluster_size_list)
                cluster_avg = np.average(cluster_size_list)

                file.write(f'{L},{T},{cluster_avg},{cluster_std}\n')
                time_for_sample = time.process_time() - start
                print(f'L = {L}, T = {T:.2f}, Cluster size = {cluster_avg:.2f}, Std = {cluster_std:.2f} --> Time for sample = {time_for_sample:.2f} seconds')

def domains_plotter():

    with open('domains.txt', 'r') as file:

        L_list = []
        T_list = []
        cluster_values = []
        cluster_error = []

        line = file.readline()
        while line:
            row = line.strip().split(',')
            L_list.append(int(row[0]))
            T_list.append(float(row[1]))
            cluster_values.append(float(row[2]))
            cluster_error.append(float(row[3]))
            line = file.readline()

        L_values = list(Counter(L_list).keys())
        T_values = list(Counter(T_list).keys())
        T_len = len(T_values)
        L_str_values = ['$T_c$']
        L_str_values.extend(['L = ' + str(L_values[i]) for i in range(len(L_values))])

        for i in range(len(L_values)-1):
            i += 1
            plt.errorbar(T_values, cluster_values[i*T_len:(i+1)*T_len], yerr = cluster_error[i*T_len:(i+1)*T_len],
                         marker = '+', label = f'L = {L_values[i]}', errorevery = 5, ecolor = 'r', capsize = 2)

        plt.axvline(x=2 / np.log(1 + np.sqrt(2)), color='k', linestyle='--', label='$T_c$')
        plt.legend()
        plt.xlabel('T (J/$k_B$)')
        plt.ylabel('Average Cluster size')
        plt.title('Average Cluster size vs Temperature')
        plt.grid()
        plt.savefig('domains1.pdf', bbox = 'tight')
        plt.show()

#simulation()
domains_plotter()
