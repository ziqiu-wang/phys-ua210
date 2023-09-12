import numpy as np
import matplotlib.pyplot as plt

mean = 0
std = 3
x_arr = np.linspace(-10, 10, 1000)
y_arr = 1/(std * np.sqrt(2*np.pi)) * np.e**(-(x_arr-mean)**2/(2*std**2))

plt.figure(1, figsize=(10, 8))
plt.plot(x_arr, y_arr, color="b")
plt.xlabel('x')
plt.ylabel('Probability density')
plt.title('Gaussian Curve')
plt.savefig("gaussian.png")
plt.show()