from random import uniform, randint
import random
from math import exp
import numpy as np
from bootstrap import bootstrap
from heat_capacity_plotter import heat_capacity_plotter
import time
import pandas as pd

# Global variables
J = 1 # Exchange energy
H = 0 # Applied magnetic field strength
mu = 1 # Magnetic moment
M = 0 # Total magnetisation
random_seed = 13
number_of_neighbours = 4
hot_start = False # Initialize with hot start
nsweeps = 60000 # Number of sweeps

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
    total_change = 2*s[index_to_flip]*sum_of_neighbours
    return total_change

def metropolis(dE_4, dE_8, N, s, neighbours_dictionary):
    """
    The metropolis algorithm as a markov chain monte carlo simulation algorithm that modifies the spin state of the
    lattice and gives a new state by choosing N (= L x L) sites at random and checking through the energy if it will
    flip the site's spin or not. The dE_4 and dE_8 are the 2 cases when the spin will be flipped and the numbers 4, 8
    represent the corresponding change in energy so that we don't calculate many times an exponential term
    :param dE_4: (float) Probability for energy change of +4
    :param dE_8: (float) Probability for energy change of +8
    :param N: (int) total number of sites
    :param s: (list) spin list
    :param neighbours_dictionary: (dict) holds indexes for neighbours
    :return: (void) does not return anything but changes the state of s
    """
    for i in range(N):
        site_index = randint(0, N-1)
        dE = get_energy_difference(site_index, s, neighbours_dictionary)
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

def get_average_energy(N, s, neighbours_dictionary):
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
    return total_energy/2*N

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

def get_average_magnetisation(N, s):
    """
    :param N: (int) total number of sites
    :param s: (list) spin list
    :return: (float) Magnetisation per site
    """
    return abs(get_magnetisation(N, s))/N

def get_autocovariance(M_list, tau):
    """
    :param M_list: (list) holding average magnetisation for each sweep neglected pre-thermalised samples
    :param tau: (int) time lag which is the input to the autocovariance formula
    :return: (float) autocovariance for the time lag tau
    """
    mean = np.average(M_list)
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
    :param autocorrelation_list: (list) autocorrelation for different tau values
    :param target_value: (float) autocorrelation initial value * 1/e
    :return: (int) index for target tau
    """
    target_index = -1
    for i in range(len(autocorrelation_list)):
        if autocorrelation_list[i] < target_value:
            target_index = i
            break
        else:
            continue
    return target_index

def get_target_tau(avg_mag_list):
    """
    :param avg_mag_list: (list) holds total magnetisation for each sweep state
    :return: (int) tau that makes autocorrelation fall to 1/e
    """
    autocorrelation_list = []
    tau_list = np.arange(0, 50, 1)

    for tau in tau_list:
        autocorrelation_list.append(get_autocorrelation(avg_mag_list, tau))

    target_value = 1/np.e
    index = get_target_value_index(autocorrelation_list, target_value)
    target_tau = tau_list[index]

    return target_tau

def simulation():
    """
    Generates data for C vs T for different values of L along with standard deviation for C using bootstrap.
    The data are written in a txt file and plotted using another file called heat_capacity_plotter.py. This
    function also finds the Tc by taking the T input that gives maximum C and prints a table of L vs Tc
    :return: (void)
    """
    T_val = np.linspace(2, 3, 100)
    T_values = [round(T_val[i], 3) for i in range(len(T_val))]
    L_values = [64]
    print(f'Total samples to calculate: {len(T_values)*len(L_values)}')
    n_bins = 100
    Tc_list = []
    thermalisation_sweeps = 10000
    sample_every = 50

    with open('heat_capacity_data1.txt', 'w') as file:
        for L in L_values:

            C_list, sigma_C_list = [], []
            N = L ** 2
            s = get_spin_list(hot_start, N)
            neighbours_dictionary = get_neighbours_index(N)

            for T in T_values:

                start = time.process_time()
                avg_mag_list = []
                energy_list = []
                b = 1 / T  # Constant: 1 / temperature
                dE_4 = exp(-4 * b)  # Probability for energy change of +4
                dE_8 = exp(-8 * b)  # Probability for energy change of +8

                for sweep in range(nsweeps):

                    metropolis(dE_4, dE_8, N, s, neighbours_dictionary)

                    if sweep < thermalisation_sweeps:
                        continue
                    else:
                        if sweep % sample_every == 0:
                            avg_mag_list.append(get_average_magnetisation(N, s))
                            energy_list.append(get_energy(N, s, neighbours_dictionary))

                target_tau = get_target_tau(avg_mag_list)
                C, sigma_C = bootstrap(energy_list, n_bins, T, target_tau)
                C_list.append(C)
                sigma_C_list.append(sigma_C)
                time_for_sample = time.process_time() - start

                file.write(f'{L},{T},{C},{sigma_C}\n')
                print(f'[L = {L}, T = {T}, C = {C:.2f}, sigma = {sigma_C:.2f}, tau = {target_tau}]', f'--> Time for sample: {time_for_sample/60:.1f} minutes')

            index_for_Tc = C_list.index(max(C_list))
            Tc_list.append(T_values[index_for_Tc])

        L_df = pd.DataFrame(L_values, columns = ['L'])
        Tc_df = pd.DataFrame(Tc_list, columns = ['Tc'])
        Tc_data = pd.concat([L_df, Tc_df], axis = 1)
        print(Tc_data)
        Tc_data.to_html('Tc_vs_L_table1.html')

simulation()
heat_capacity_plotter()
