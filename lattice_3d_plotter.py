import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def lattice_3d_plotter():

    with open('lattice_3d.txt', 'r') as file:

        onsager_Tc = 2 / np.log(1 + np.sqrt(2))

        L_list = []
        T_list = []
        avg_mag_list = []
        avg_energy_list = []

        line = file.readline()
        while line:
            row = line.strip().split(',')
            L_list.append(int(row[0]))
            T_list.append(float(row[1]))
            avg_mag_list.append(float(row[2]))
            avg_energy_list.append(float(row[3]))
            line = file.readline()

        L_values = list(Counter(L_list).keys())
        L_length = len(L_values)
        T_values = list(Counter(T_list).keys())
        T_length = len(T_values)

        L_str_values = ['L = ' + str(L_values[i]) for i in range(len(L_values))]
        L_str_values.insert(0, 'Onsager\'s $T_c$')

        for i in range(L_length):
            plt.scatter(T_values, avg_mag_list[i*T_length:(i+1)*T_length], marker = '+')

        plt.axvline(x=onsager_Tc, color='k', linestyle='--')
        plt.legend(L_str_values)
        # plt.subplots_adjust(left=0.15, right=0.98)
        plt.title(f'Magnetisation per site vs temperature')
        plt.xlabel('T (J/$k_B$)')
        plt.ylabel('<m>')
        plt.savefig('lattice_3d_mag.pdf', bbox='tight')
        plt.show()

        for i in range(L_length):
            plt.scatter(T_values, avg_energy_list[i*T_length:(i+1)*T_length], marker = '+')

        plt.axvline(x=onsager_Tc, color='k', linestyle='--')
        plt.legend(L_str_values)
        # plt.subplots_adjust(left=0.15, right=0.98)
        plt.title(f'Energy per site vs temperature')
        plt.xlabel('T (J/$k_B$)')
        plt.ylabel('<E>')
        plt.savefig('lattice_3d_energy.pdf', bbox='tight')
        plt.show()
