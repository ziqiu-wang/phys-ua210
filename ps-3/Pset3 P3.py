import numpy as np
import matplotlib.pyplot as plt
from random import random

#numbers of atoms
n_Bi209 = 0
n_Pb = 0
n_Tl = 0
n_Bi213 = 10000

#time interval, max time, and half-lives
dt = 1
tmax = 20000
tau_Pb = 3.3 * 60
tau_Tl = 2.2 * 60
tau_Bi = 46 * 60

#decay probability
p_Pb = 1 - 2**(-dt/tau_Pb)
p_Tl = 1 - 2**(-dt/tau_Tl)
p_Bi = 1 - 2**(-dt/tau_Bi)

#loop for 20000 seconds
Bi209 = [n_Bi209]
Pb = [n_Pb]
Tl = [n_Tl]
Bi213 = [n_Bi213]

for i in range(tmax):
    #Pb
    decay = 0
    for j in range(n_Pb):
        if random() < p_Pb:
            decay += 1
    n_Pb -= decay
    n_Bi209 += decay
    #Tl
    decay = 0
    for j in range(n_Tl):
        if random() < p_Tl:
            decay += 1
    n_Tl -= decay
    n_Pb += decay
    #Bi213
    decay = 0
    decay_to_Pb = 0
    decay_to_Tl = 0
    for j in range(n_Bi213):
        if random() < p_Bi:
            decay += 1
    for j in range(decay):
        if random() < 0.9791:
            decay_to_Pb += 1
        else:
            decay_to_Tl += 1
    n_Bi213 -= decay
    n_Pb += decay_to_Pb
    n_Tl += decay_to_Tl
    #appending points to the lists
    Bi209.append(n_Bi209)
    Pb.append(n_Pb)
    Tl.append(n_Tl)
    Bi213.append(n_Bi213)

#plotting
times = np.arange(0,tmax+1,dt)
Bi209 = np.array(Bi209)
Pb = np.array(Pb)
Tl = np.array(Tl)
Bi213 = np.array(Bi213)

plt.plot(times, Bi209)
plt.plot(times, Pb)
plt.plot(times, Tl)
plt.plot(times, Bi213)
plt.legend(["Bi209", "Pb", "Tl", "Bi213"])
plt.ylabel("number of atoms")
plt.xlabel("time (s)")
plt.title("Number of Atoms vs. Time")
plt.show()




