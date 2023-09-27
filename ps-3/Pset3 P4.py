import numpy as np
import matplotlib.pyplot as plt

z = np.random.uniform(size=1000)
tau = 60 * 3.053
t = -1/(np.log(2)/tau) * np.log(1-z) #1000 decay times
sorted_t = np.sort(t) #sorting
sorted_t = np.concatenate(([0], sorted_t)) #insert a 0 at the beginning
not_decay = np.arange(1000,-1,-1) #from 1000 to 0 (included)
print(not_decay)

#plotting
plt.plot(sorted_t, not_decay, "olive")
plt.ylabel("number of atoms that have not decayed")
plt.xlabel("time (s)")
plt.title('Number of Atoms That Have not Decayed vs. Time')
plt.show()