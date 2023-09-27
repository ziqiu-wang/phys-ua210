import numpy as np
import matplotlib.pyplot as plt

#Part (a)
def f(x):
    return x*(x - 1)

def f_prime(x):
    return 2*x - 1

x = 1
delta = 1e-2
deriv_rough = (f(x + delta) - f(x))/delta
print("Approximation:", deriv_rough)
deriv_accurate = f_prime(x)
print("Accurate result:", deriv_accurate)

#Part (b)
derivs = []
for i in range(6):
    delta *= 1e-2
    derivs.append((f(x + delta) - f(x))/delta)
print(derivs)

log_x = np.array([-4, -6, -8, -10, -12, -14])
log_errs = np.log10(np.abs(np.array(derivs) - deriv_accurate))

plt.scatter(log_x, log_errs, color="b")
plt.ylabel('log of absolute error (base 10)')
plt.xlabel('$\log_{10}(\delta)$')
plt.title('Logs of Absolute Errors in the Approximations')
plt.show()





