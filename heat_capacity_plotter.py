import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

def heat_capacity_plotter():
    """
    Plots C/N vs T for different values of L
    :return: (void)
    """
    with open('heat_capacity_data1.txt', 'r') as file:

        onsager_Tc = 2/np.log(1+np.sqrt(2))

        L_list = []
        T_list = []
        C_list = []
        sigma_C_list = []

        line = file.readline()
        while line:
            row = line.strip().split(',')
            L_list.append(int(row[0]))
            T_list.append(float(row[1]))
            C_list.append(float(row[2]))
            sigma_C_list.append(float(row[3]))
            line = file.readline()

        L_values = list(Counter(L_list).keys())
        L_length = len(L_values)
        T_length = len(list(Counter(T_list).keys()))

        L_str_values = ['L = ' + str(L_values[i]) for i in range(len(L_values))]
        L_str_values.insert(0, 'Onsager\'s $T_c$')

        C_array = np.array(C_list)
        sigma_C_array = np.array(sigma_C_list)
        N_values_array = np.array(L_values)**2

        colour_list = ['k', 'b', 'g', 'y', 'm', 'c', 'r', 'navy', 'lightcoral', 'lime']

        for i in range(L_length):
            plt.errorbar(T_list[i*T_length:(i+1)*T_length], C_array[i*T_length:(i+1)*T_length]/N_values_array[i],
                         xerr = 0, yerr = sigma_C_array[i*T_length:(i+1)*T_length]/N_values_array[i], ls = '',
                         marker = '+', ecolor = 'r', capsize = 2, color = colour_list[i], elinewidth = 0.7, errorevery = 30)


        plt.axvline(x=onsager_Tc, color='k', linestyle='--')
        # plt.yscale('log')
        plt.legend(L_str_values)
        plt.title('Specific heat capacity per site vs temperature')
        plt.xlabel('T (J/$k_B$)')
        plt.ylabel('C/N')
        # plt.savefig('heat_cap_vs_tmp2.pdf', bbox = 'tight')
        plt.show()

        tc_fitter(L_list, T_list, C_list, sigma_C_list)

def gaussian(x, mu, sigma, amp):
    return amp*np.exp(-((x-mu)/sigma)**2)

def tc_fitter(L_values, T_values, C_values, sigma_C_values):

    # L_values = [4, 8, 10, 12, 24, 32, 48]
    # tc_values = [2.394, 2.354, 2.364, 2.384, 2.293, 2.293, 2.273]

    L_val = list(Counter(L_values).keys())
    L_len = len(L_val)
    T_len = len(list(Counter(T_values).keys()))

    tc_list = []
    tc_error = []

    with open('finite_scaling_data.txt', 'w') as file:
        for i in range(L_len):

            T_tmp = T_values[i*T_len:(i+1)*T_len]
            C_tmp = C_values[i*T_len:(i+1)*T_len]
            sigma_C_tmp = sigma_C_values[i*T_len:(i+1)*T_len]
            index_of_max = C_tmp.index(max(C_tmp))
            T = T_tmp[index_of_max - 20:index_of_max + 20]
            C = C_tmp[index_of_max - 20:index_of_max + 20]
            C_sigma = sigma_C_tmp[index_of_max - 20:index_of_max + 20]
            parameters, pcov = curve_fit(gaussian, T, C, sigma = C_sigma, absolute_sigma = True)
            mu, sigma, amp, sigma_mu = parameters[0], parameters[1], parameters[2], np.sqrt(np.diag(pcov)[0])
            tc_list.append(mu)
            tc_error.append(sigma_mu)
            file.write(f'{L_val[i]**2},{mu},{sigma_mu}\n')

    L_df = pd.DataFrame(L_val, columns = ['L'])
    tc_df = pd.DataFrame(tc_list, columns = ['Tc'])
    tc_error_df = pd.DataFrame(tc_error, columns = ['sigma_Tc'])
    df = pd.concat([L_df, tc_df, tc_error_df], axis = 1)
    # df.to_html('L_Tc_error.html')
    print(df)


def func(N, tc_at_inf, a, nu):
    return tc_at_inf + a*N**(-1/nu)

def finite_scaling_fitter():

    with open('finite_scaling_data.txt', 'r') as file:

        N_list = []
        tc_list = []
        tc_error = []

        line = file.readline()
        while line:
            row = line.strip().split(',')
            N_list.append(float(row[0]))
            tc_list.append(float(row[1]))
            tc_error.append(float(row[2]))
            line = file.readline()

        p0 = [2.26, 1.0, 2.0]

        parameters, covariance_matrix = curve_fit(func, N_list, tc_list,
                                                  sigma = tc_error, absolute_sigma = True, p0 = p0)
        tc_at_inf, a, nu = parameters[0], parameters[0], parameters[0]
        tc_at_inf_error, a_error, nu_error = np.sqrt(np.diag(covariance_matrix)[0]), \
                                             np.sqrt(np.diag(covariance_matrix)[1]),\
                                             np.sqrt(np.diag(covariance_matrix)[2])

        print(f'\nTc at infinity: {tc_at_inf:.3f}, sigma: {tc_at_inf_error:.3f}\na: {a:.3f}, '
              f'sigma: {a_error:.3f}\nnu: {nu:.3f}, sigma: {nu_error:.3f}')

        plt.plot(N_list, tc_list, color = 'k', linestyle = '--')
        plt.plot(N_list, tc_at_inf + a*np.array(N_list)**(-1/nu), color = 'r')
        plt.title('Finite-size scaling: $T_c$ vs N\nFit: $T_c(N)$ = $T_c(\u221e) + \u03B1(N)^{-\\frac{1}{\u03BD}}$')
        plt.ylabel('$T_c(N)$')
        plt.xlabel('N')
        plt.legend(['Simulation', 'Fit'])
        plt.savefig('finite_size_scaling.pdf', bbox = 'tight')
        plt.show()

heat_capacity_plotter()
finite_scaling_fitter()
