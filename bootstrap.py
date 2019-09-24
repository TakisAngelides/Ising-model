import random
import numpy as np

def get_sigma_energy(pseudo_bin_list):
    """
    :param pseudo_bin_list: (list) contains the pseudo data of one bin
    :return: (float) the standard deviation of the input list
    """
    return np.std(pseudo_bin_list)

def get_heat_capacity(sigma_energy, T):
    """
    :param sigma_energy: (float) the standard deviation of the energy from a bin list
    :param T: (float) Temperature
    :return: (float) the specific heat capacity
    """
    return (sigma_energy**2) / (T**2)

def bootstrap(energy_list, n_bins, T, target_tau):
    """
    :param target_tau: (int) decorrelation constant
    :param energy_list: (list) energy for each sweep
    :param n_bins: (int) number of bins to include in the boostrap method
    :param T: (float) Temperature
    :return: (float, float) 2-tuple of heat capacity and its standard deviation
    """
    length = len(energy_list)
    # Initialise the bin dictionary holding values of lists. These lists will contain pseudo data of energy
    bins_dict = {idx: [] for idx in range(n_bins)}
    # This for loop traverses the bin keys in the dictionary bins_dict
    for bin_index in range(n_bins):
        # This for loop runs for the length of the energy list
        for i in range(length):
            random_index = random.randint(0, length-1)
            bins_dict[bin_index].append(energy_list[random_index]) # Append pseudo data to bin list

    sigma_energy_list = [get_sigma_energy(bins_dict[key_bin]) for key_bin in bins_dict]
    C_list = [get_heat_capacity(sigma_energy_list[i], T) for i in range(len(sigma_energy_list))]

    C_mean = np.average(C_list)
    sigma_C = np.std(C_list)*np.sqrt(1 + 2*target_tau)

    return C_mean, sigma_C





