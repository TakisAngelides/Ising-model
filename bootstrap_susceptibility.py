import random
import numpy as np

def get_sigma_magnetisation(pseudo_bin_list):
    """
    :param pseudo_bin_list: (list) contains the pseudo data of one bin
    :return: (float) the standard deviation of the input list
    """
    return np.std(pseudo_bin_list)

def get_susceptibility(sigma_magnetisation, T):
    """
    :param sigma_magnetisation: (float) the standard deviation of the magnetisation from a bin list
    :param T: (float) Temperature
    :return: (float) the susceptibility
    """
    return (sigma_magnetisation**2) / (T**2)

def bootstrap(magnetisation_list, n_bins, T, target_tau):
    """
    :param target_tau: (int) decorrelation constant
    :param magnetisation_list: (list) magnetisation for each sweep
    :param n_bins: (int) number of bins to include in the boostrap method
    :param T: (float) Temperature
    :return: (float, float) 2-tuple of susceptibility and its standard deviation
    """
    length = len(magnetisation_list)
    # Initialise the bin dictionary holding values of lists. These lists will contain pseudo data of magnetisation
    bins_dict = {idx: [] for idx in range(n_bins)}
    # This for loop traverses the bin keys in the dictionary bins_dict
    for bin_index in range(n_bins):
        # This for loop runs for the length of the magnetisation list
        for i in range(length):
            random_index = random.randint(0, length-1)
            bins_dict[bin_index].append(magnetisation_list[random_index]) # Append pseudo data to bin list

    sigma_magnetisation_list = [get_sigma_magnetisation(bins_dict[key_bin]) for key_bin in bins_dict]
    chi_list = [get_susceptibility(sigma_magnetisation_list[i], T) for i in range(len(sigma_magnetisation_list))]

    chi_mean = np.average(chi_list)
    sigma_chi = np.std(chi_list)*np.sqrt(1 + 2*target_tau)

    return chi_mean, sigma_chi





