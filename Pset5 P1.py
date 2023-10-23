import numpy as np
import matplotlib.pyplot as plt

# Part (a)
def integrand(x,a):
    return x**(a-1) * np.exp(-x)

x = np.arange(0,5,0.02)
y2 = integrand(x,2)
y3 = integrand(x,3)
y4 = integrand(x,4)

# Plotting
plt.plot(x, y2, "olive")
plt.plot(x, y3, "b")
plt.plot(x, y4, "r")
plt.xlabel("$x$")
plt.ylabel("Integrand")
plt.legend(["$a=2$", "$a=3$", "$a=4$"])
plt.title("Integrand vs. x")
plt.show()

# Part (e)
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

# Integrand
def I(a,z):
    c = a-1
    expo = np.exp(c*np.log(z*c/(1-z))-z*c/(1-z))
    other_term = (a-1)/(1-z)**2
    return expo * other_term

# Function that calculates integral
def gamma(a):
    z,w = gaussxwab(100,0,1)
    s = 0.0
    for k in range(100):
        s += w[k] * I(a,z[k])
    return s

# Computing integral
low = 0
hi = 1
a = 3.0/2
result = gamma(a)

print(result)
print(result - 1/2*np.sqrt(np.pi))

# Part (f)
gamma3 = gamma(3)
gamma6 = gamma(6)
gamma10 = gamma(10)

print(gamma3)
print(gamma6)
print(gamma10)




