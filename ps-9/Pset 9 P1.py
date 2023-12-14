import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

m = 9.109e-31
L = 1e-8
x0 = L/2
sigma = 1e-10
k = 5e10
N = 1000
a = L/N
h = 1e-18
h_bar = 1.054571817e-34

a1 = 1 + h * 1j * h_bar / (2 * m * a**2)
a2 = -h * 1j * h_bar / (4 * m * a**2)
b1 = 1 - h * 1j * h_bar / (2 * m * a**2)
b2 = -a2

# Vector Psi(0) at t = 0
psi = np.zeros(N+1, dtype=np.complex64)   # 0, a, 2a, 3a,..., L=na
for n in range(N-1):
    x = a * (n + 1)
    psi[n+1] = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k * x)

# Matrices A and B
A = np.zeros((N+1, N+1), dtype=np.complex64)
B = np.zeros((N+1, N+1), dtype=np.complex64)
for i in range(N+1):
    for j in range(N+1):
        if i == j:
            A[i][j] = a1
            B[i][j] = b1
        elif abs(i-j) == 1:
            A[i][j] = a2
            B[i][j] = b2
A_inv = np.linalg.inv(A)

# Solve the problem
num_steps = 5000
psi_lst = [psi]
for _ in range(num_steps):
    v = np.matmul(B, psi)
    psi = np.matmul(A_inv, v)   # equivalent to using psi = np.linalg.solve(A,v)
    psi_lst.append(psi)
psi_arr = np.array(psi_lst)
print(len(psi_arr))

# Plot the wavefunctions (unnormalized)
x = np.linspace(0, L * 1e9, 1001)    # work in units of nm
fig, ax = plt.subplots(1, 1)

'''
The following is an animated version of:

for p in psi_arr:
    plt.plot(x, p)
    plt.ylim(-1, 1)
    plt.xlabel("$x$ (nm)")
    plt.ylabel("$\psi$")
    plt.show()
'''

def animate(i):
    ax.clear()
    p = psi_arr[i]
    plt.plot(x, p)
    plt.ylim(-1, 1)
    plt.xlabel("$x$ (nm)")
    plt.ylabel("$\psi$")

ani = FuncAnimation(fig, animate, frames=num_steps + 1, repeat=False)
ani.save("wavefunction.gif", dpi=300, writer=PillowWriter(fps=30))


