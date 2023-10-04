import numpy as np
import matplotlib.pyplot as plt

def gaussxw(N):
    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3,4*N-1,N)/(4*N+2)
    x = np.cos(np.pi*a+1/(8*N*N*np.tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = np.ones(N,float)
        p1 = np.copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a), 0.5*(b-a)*w

# Potential function
def V(x):
    return x**4

# Integrand
def f(x,a):
    return 1/np.sqrt(V(a)-V(x))

# Function to calculate the period
# low is lower bound, a is higher bound
def period(N,low,a,m):
    x,w = gaussxwab(N,low,a)
    T = 0.0
    for k in range(N):
        T += w[k] * f(x[k],a)
    T *= np.sqrt(8*m)
    return T

# Function calls for a between 0 and 2
N = 20
low = 0.0
m = 1
T = []
a_arr = np.linspace(0.0, 2.0, 101)[1:]
for a in a_arr:
    T.append(period(N,low,a,m))
T_arr = np.array(T)

# Plotting
plt.plot(a_arr, T_arr, "olive")
plt.ylabel("$T$")
plt.xlabel("$a$")
plt.title("Period vs. Amplitude")
plt.show()

