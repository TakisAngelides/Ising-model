import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def nearest_neighbours_plotter():

    with open('nearest_neighbours1.txt', 'r') as file:

        onsager_Tc = 2 / np.log(1 + np.sqrt(2))

        L_list = []
        J_list = []
        T_list = []
        avg_mag_list = []

        line = file.readline()
        while line:
            row = line.strip().split(',')
            L_list.append(int(row[0]))
            J_list.append(float(row[1]))
            T_list.append(float(row[2]))
            avg_mag_list.append(float(row[3]))
            line = file.readline()

        L_values = list(Counter(L_list).keys())
        L_length = len(L_values)
        J_values = list(Counter(J_list).keys())
        J_length = len(J_values)
        T_values = list(Counter(T_list).keys())
        T_length = len(T_values)

        J_str_values = ['J = ' + str(J_values[i]) for i in range(len(J_values))]
        J_str_values.insert(0, 'Onsager\'s $T_c$')

        for i in range(J_length):
            plt.scatter(T_values, avg_mag_list[i*T_length:(i+1)*T_length], marker = '+')

        plt.axvline(x=onsager_Tc, color='k', linestyle='--')
        plt.legend(J_str_values)
        # plt.subplots_adjust(left=0.15, right=0.98)
        plt.title(f'Magnetisation per site vs temperature (L = {L_values[0]})')
        plt.xlabel('T (J/$k_B$)')
        plt.ylabel('<m>')
        plt.savefig('nearest_neighbours1.pdf', bbox='tight')
        plt.show()

nearest_neighbours_plotter()