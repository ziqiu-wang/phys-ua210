import numpy as np
import matplotlib.pyplot as plt
import math
import scipy

# Part (a)
def H(n,x):
    # Non-recursive method using memoization
    if n == 0:
        return 1
    elif n == 1:
        return 2*x
    memo = [1, 2*x]
    for i in range(n-1):
        memo.append(2*x*memo[i+1] - 2*(i+1)*memo[i])
    return memo[-1]

def H_rec(n,x):
    # Recursive method, not used
    if n > 1:
        return 2*x*H(n-1,x) - 2*(n-1)*H(n-2,x)
    elif n == 1:
        return 2*x
    elif n == 0:
        return 1

def wavef(n,x):
    # Wave function
    return 1/np.sqrt(2**n * math.factorial(n) * np.sqrt(np.pi)) \
           * np.exp(-x**2/2) * H(n,x)

x = np.linspace(-4.0, 4.0, 101)
for n in [0,1,2,3]:
    psi = wavef(n,x)
    plt.plot(x,psi)
plt.ylabel("$\psi(x)$")
plt.xlabel("$x$")
plt.legend(["n=0","n=1","n=2","n=3"])
plt.title("Wavefunctions with different $n$")
plt.show()

# Part (b)
x = np.linspace(-10, 10, 201)
n = 30
psi = wavef(n,x)
plt.plot(x,psi,"olive")
plt.ylabel("$\psi(x)$")
plt.xlabel("$x$")
plt.legend(["n=30"])
plt.title("Wavefunction with $n=30$")
plt.show()

# Part (c)
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

def f(n,x):
    return x**2 * wavef(n,x)**2

# Integrand
def I(n,z):
    return f(n,np.tan(z)) / (np.cos(z)**2)

# Function that calculates integral
def integrate(low,hi,n):
    z,w = gaussxwab(100,low,hi)
    s = 0.0
    for k in range(100):
        s += w[k] * I(n,z[k])
    return s

# Calculating integral for n = 5
low = -np.pi/2
hi = np.pi/2
n = 5
result = integrate(low,hi,n)
rmsq = np.sqrt(result)
print(rmsq)

# Part (d)
def g(n,x):
    # exponential term exp(-x^2) is omitted here
    return x**2 * 1/(2**n * math.factorial(n) * np.sqrt(np.pi)) * (H(n,x))**2

def integrateGH(n):
    x,w = np.polynomial.hermite.hermgauss(100)
    s = 0.0
    for k in range(100):
        s += w[k] * g(n,x[k])
    return s

n = 5
GH_result = integrateGH(n)
rmsq_GH = np.sqrt(GH_result)
print(rmsq_GH)









