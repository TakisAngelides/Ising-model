from random import uniform, randint
import random
from math import exp
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras.layers import Conv2D
from hyperopt import hp


# Global variables
J = 1 # Exchange energy
L = 10 # Lattice dimension
H = 0 # Applied magnetic field strength
mu = 1 # Magnetic moment
M = 0 # Total magnetisation
number_of_neighbours = 4
hot_start = True # Initialize with hot start or not
nsweeps = 500 # Number of sweeps

def get_spin_list(is_hot, N):
    """
    :param is_hot: (bool) If is_hot is true make hot start otherwise make cold start
    :param N: (int) N = L x L, total number of sites
    :return: (list) containing +/- 1 representing the spins on the lattice
    """
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
    total_change = 2*s[index_to_flip]*sum_of_neighbours # Works out from the Hamiltonian of the before and after states
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
        site_index = randint(0,N-1)
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

def simulation():

    with open('deep_ising_data.txt', 'w') as f:
        # Temperature values in Kelvin
        T_min = 1.8
        T_max = 6
        N = L**2
        s = get_spin_list(hot_start, N)
        neighbours_dictionary = get_neighbours_index(N)
        number_of_runs = 50000
        configurations = []
        for i in range(number_of_runs):
            T = round(random.uniform(T_min, T_max), 2)
            b = 1 / T  # Constant: 1 / temperature
            dE_4 = exp(-4 * b)  # Probability for energy change of +4
            dE_8 = exp(-8 * b)  # Probability for energy change of +8
            for j in range(nsweeps):
                metropolis(dE_4, dE_8, N, s, neighbours_dictionary)
            configurations.append(s + [T])
            print(f'Temp: {T} | run: {i} done')
            str_temp = ','.join(map(str, s+[T]))
            f.write(f'{str_temp}\n')

def get_model():
    model = Sequential()
    model.add(Dense(L*L, activation='sigmoid', input_shape=(L*L,)))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    return model

def get_data():

    with open('deep_ising_data.txt', 'r') as f:
        X = []
        Y = []
        line = f.readline()
        while line:
            configuration = line.strip().split(',')
            X.append([int(configuration[:-1][i]) for i in range(len(configuration[:-1]))])
            Y.append(float(configuration[-1]))
            line = f.readline()
        X_train = X[:int(len(X)*0.8)]
        X_test = X[int(len(X)*0.8):]
        Y_train = Y[:int(len(Y)*0.8)]
        Y_test = Y[int(len(Y)*0.8):]

    return np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)

model = load_model('ising_network.h5')
# X_train, X_test, Y_train, Y_test = get_data()
# model = get_model()
# history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs = 20)
# plt.plot(history.history['loss'], color = 'k')
# plt.plot(history.history['val_loss'], color = 'r')
# plt.legend(['MSE', 'Validation MSE'])
# plt.title('Neural network prediction performance on temperature')
# plt.xlabel('Epoch')
# plt.ylabel('Mean squared error')
# plt.savefig('deep_ising_mse.pdf', bbox_inches = 'tight')
# plt.show()
# model.save('ising_network.h5')
# model.summary()

def thermalize(T):
    N = L**2
    s = get_spin_list(hot_start, N)
    neighbours_dictionary = get_neighbours_index(N)
    b = 1 / T  # Constant: 1 / temperature
    dE_4 = exp(-4 * b)  # Probability for energy change of +4
    dE_8 = exp(-8 * b)  # Probability for energy change of +8
    for _ in range(nsweeps):
        metropolis(dE_4, dE_8, N, s, neighbours_dictionary)

    return np.array([s])

predictions = [round(float(model.predict(thermalize(1.9))), 2) for _ in range(5000)]
plt.axvline(x = 1.9, linestyle = '--', color = 'r')
plt.hist(predictions, color = 'k', bins = 59)
plt.title('Histogram for DNN prediction on T = 1.9')
plt.ylabel('Frequency of prediction')
plt.xlabel('Temperature')
plt.savefig('deep_histogram2.pdf', bbox_inches = 'tight')
plt.show()
