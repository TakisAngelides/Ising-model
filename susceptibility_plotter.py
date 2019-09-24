import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def susceptibility_plotter():
    """
    Plots chi/N vs T for different values of L
    :return: (void)
    """
    with open('susceptibility.txt', 'r') as file:

        onsager_Tc = 2/np.log(1+np.sqrt(2))

        L_list = []
        T_list = []
        chi_list = []
        sigma_chi_list = []

        line = file.readline()
        while line:
            row = line.strip().split(',')
            L_list.append(int(row[0]))
            T_list.append(float(row[1]))
            chi_list.append(float(row[2]))
            sigma_chi_list.append(float(row[3]))
            line = file.readline()

        L_values = list(Counter(L_list).keys())
        L_length = len(L_values)
        T_length = len(list(Counter(T_list).keys()))

        L_str_values = ['L = ' + str(L_values[i]) for i in range(len(L_values))]
        L_str_values.insert(0, 'Onsager\'s $T_c$')

        chi_array = np.array(chi_list)
        sigma_chi_array = np.array(sigma_chi_list)
        N_values_array = np.array(L_values)**2

        colour_list = ['k', 'b', 'g', 'y', 'm', 'c', 'r', 'navy', 'lightcoral', 'lime']

        for i in range(L_length):

            plt.errorbar(T_list[i*T_length:(i+1)*T_length], chi_array[i*T_length:(i+1)*T_length]/N_values_array[i],
                         xerr = 0, yerr = sigma_chi_array[i*T_length:(i+1)*T_length]/N_values_array[i], ls = '',
                         marker = '+', ecolor = 'r', capsize = 2, color = colour_list[i], elinewidth = 0.7, errorevery = 1)

        plt.axvline(x=onsager_Tc, color='k', linestyle='--')
        # plt.yscale('log')
        plt.legend(L_str_values)
        plt.subplots_adjust(left = 0.15, right = 0.98)
        plt.title('Susceptibility per site vs temperature')
        plt.xlabel('T (J/$k_B$)')
        plt.ylabel('$\u03C7$/N')
        plt.savefig('susceptibility.pdf', bbox = 'tight')
        plt.show()

susceptibility_plotter()
