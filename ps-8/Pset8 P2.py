import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def fun(t,v,s,r,b):
    x, y, z = v
    return [s*(y-x), r*x-y-x*z, x*y-b*z]

sol = solve_ivp(fun, [0, 50], [0,1,0], args=(10,28,8/3), dense_output=True)
# noinspection PyUnresolvedReferences
t, v = sol.t, sol.y
print(t)
print(v)
x, y, z = v[0], v[1], v[2]

plt.plot(t,y)
plt.xlabel("$t$")
plt.ylabel("$y$")
plt.title("Solution for $y(t)$")
plt.show()

plt.plot(x,z)
plt.xlabel("$x$")
plt.ylabel("$z$")
plt.title("Plot of $z(t)$ versus $x(t)$")
plt.show()