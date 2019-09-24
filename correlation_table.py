from random import uniform, randint
import random
from math import exp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import time

# Global variables
J = 1 # Exchange energy
H = 0 # Applied magnetic field strength
mu = 1 # Magnetic moment
M = 0 # Total magnetisation
number_of_neighbours = 4
random_seed = 10 # Fix random seed in spin initialization for reproducability
hot_start = True # Initialize with hot start
nsweeps = 75000 # Number of sweeps

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

def get_neighbours_index(L):
    """
    :param L: (int) Dimension of the lattice
    :return: (dict) containing as keys the index of the site on the lattice and as values a list containing the indexes
    of its neighbours
    """
    neighbours_dict = {}
    N = L**2
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

def get_energy_difference(s, neighbours_dictionary, index_to_flip):
    """
    :param s: (list) spin list of the lattice
    :param neighbours_dictionary: (dict) holds indexes for each site's neighbours
    :param index_to_flip: (int) the site index to consider flipping its spin
    :return: (float) Total energy change of changing that site's spin
    """
    sum_of_neighbours = 0
    for neighbour_index in neighbours_dictionary[index_to_flip]:
        sum_of_neighbours += s[neighbour_index]
    total_change = 2*s[index_to_flip]*sum_of_neighbours
    return total_change

def metropolis(s, neighbours_dictionary, N, dE_4, dE_8):
    """
    The metropolis algorithm as a markov chain monte carlo simulation algorithm that modifies the spin state of the
    lattice and gives a new state by choosing N (= L x L) sites at random and checking through the energy if it will
    flip the site's spin or not. The dE_4 and dE_8 are the 2 cases when the spin will be flipped and the numbers 4, 8
    represent the corresponding change in energy so that we don't calculate many times an exponential term
    :param s: (list) spin list
    :param neighbours_dictionary: (dict) holds indexes for neighbours
    :param N: (int) total number of sites
    :param dE_4: (float) Probability for energy change of +4
    :param dE_8: (float) Probability for energy change of +8
    :return: (void) does not return anything but changes the state of s
    """
    for i in range(N):
        site_index = randint(0,N-1)
        dE = get_energy_difference(s, neighbours_dictionary, site_index)
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

def get_magnetisation(s, N):
    """
    :param s: (list) spin list
    :param N: (int) total number of sites
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
            sum2 = get_magnetisation(s, N)
    total_energy = (-J*sum1 - mu*H*sum2)/2
    return total_energy

def get_average_magnetisation(s, N):
    """
    :param s: (list) spin list
    :param N: (int) total number of sites
    :return: (float) Magnetisation per site
    """
    return abs(get_magnetisation(s, N))/N

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

def get_target_tau(s, L, dE_4, dE_8, neighbours_dictionary):
    """
    :param s: (list) spin list
    :param L: (int) lattice dimension
    :param dE_4: (float) Probability for energy change of +4
    :param dE_8: (float) Probability for energy change of +8
    :param neighbours_dictionary: (dict) holds indexes for neighbours
    :return: (int) tau that makes autocorrelation fall to 1/e
    """
    N = L**2
    avg_magnetisation_list = []
    autocorrelation_list = []
    tau_list = np.arange(0, 2500, 1)
    thermalisation_sweeps = 20000
    sample_every = 1

    for sweep in range(nsweeps):

        metropolis(s, neighbours_dictionary, N, dE_4, dE_8)

        if sweep < thermalisation_sweeps:
            continue
        else:
            if sweep % sample_every == 0:
                avg_magnetisation_list.append(get_average_magnetisation(s, N))
            else:
                continue

    for tau in tau_list:
        autocorrelation_list.append(get_autocorrelation(avg_magnetisation_list, tau))

    target_value = 1/np.e
    index = get_target_value_index(autocorrelation_list, target_value)
    target_tau = tau_list[index]

    return target_tau

def simulation():
    """
    For different values of L and T the simulation finds tau that makes autocorrelation fall to 1/e.
    It stores these data in a dataframe and prints an html table of it
    :return: (void)
    """
    tmp_data_list = []
    T_val = [2.3]
    T_values = [round(T_val[i], 3) for i in range(len(T_val))]
    L_values = [10, 20, 30, 40, 50]
    print(f'Total number of samples: {len(L_values)*len(T_values)}')

    with open('correlation_table.txt', 'w') as file:
        for L in L_values:

            N = L**2
            s = get_spin_list(hot_start, N)
            neighbours_dictionary = get_neighbours_index(L)

            for T in T_values:

                start = time.process_time()
                b = 1 / T  # Constant: 1 / temperature
                dE_4 = exp(-4 * b)  # Probability for energy change of +4
                dE_8 = exp(-8 * b)  # Probability for energy change of +8
                target_tau = get_target_tau(s, L, dE_4, dE_8, neighbours_dictionary)
                # tmp_data_list.append([N, T, target_tau])
                time_for_sample = time.process_time() - start
                print(f'[L = {L}, T = {T}, tau = {target_tau}] --> Time for sample: {time_for_sample/60:.1f} minutes')
                file.write(f'{L},{T},{target_tau}\n')

def function_to_fit(L_array, z, a, b):
    """
    :param L_array: (nparray) Contains the L values tried
    :param z: (float) z exponent in critical slowing down equation
    :param a: (float) constant of equation
    :param b: (float) constant of equation
    :return: (nparray) The function to fit by finding z, a, b
    """
    return a*(L_array**z)+b

def critical_slowing_down_fit():
    """
    Tries to fit the data of tau vs L to find the z exponent in the critical slowing down relationship.
    It also makes a plot of the data and the fit and prints z
    :return: (void)
    """
    with open('correlation_table.txt', 'r') as file:

        L_list = []
        T_list = []
        tau_list = []

        line = file.readline()
        while line:
            row = line.strip().split(',')
            L_list.append(int(row[0]))
            T_list.append(float(row[1]))
            tau_list.append(float(row[2]))
            line = file.readline()

        L_list_df = pd.DataFrame(L_list, columns = ['L'])
        T_list_df = pd.DataFrame(T_list, columns = ['Temperature'])
        tau_list_df = pd.DataFrame(tau_list, columns = ['tau'])

        df = pd.concat([L_list_df, T_list_df, tau_list_df], axis = 1)
        df.to_html('L_T_tau2.html')
        print(df)

        tau_array = np.array(tau_list)
        L_array = np.array(L_list)

        p0 = [2.1, 0.2, 1]
        param_fit, cov = curve_fit(function_to_fit, L_array, tau_array, p0 = p0)
        print(f'z was found to be: {param_fit[0]}, with standard deviation: {np.sqrt(np.diag(cov)[0])}')
        z_fit, a_fit, b_fit = param_fit[0], param_fit[1], param_fit[2]
        tau_fit = a_fit*(L_array**z_fit) + b_fit

        plt.scatter(L_array, tau_array, marker = '+', color = 'r', s = 100)
        plt.plot(L_array, tau_fit, color = 'k')
        plt.legend([f'Fit: z = {z_fit:.3f}', 'Simulation'])
        plt.title('Critical slowing down $\u03C4_e$ vs L\nFit: $\u03C4_e$$\sim$$L^{z}$')
        plt.xlabel('L')
        plt.ylabel('$\u03C4_e$')
        plt.grid(True)
        # plt.savefig('critical_slowing_down.pdf', bbox = 'tight')
        plt.show()

critical_slowing_down_fit()
