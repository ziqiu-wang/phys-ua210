import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def prob(x, b0, b1):
    return 1/(1+np.exp(-(b0 + b1*x)))

def neg_log_like(b, x, y):
    # negative log likelihood given beta_0 and beta_1
    mach_prec = 1e-16  # avoid divide by 0 error
    b0 = b[0]
    b1 = b[1]
    l_list = [y[i] * np.log(prob(x[i], b0, b1)/(1-prob(x[i], b0, b1) + mach_prec))
              + np.log(1-prob(x[i], b0, b1) + mach_prec) for i in range(len(x))]
    ll = np.sum(np.array(l_list))
    return -ll

# Covariance matrix of parameters
def covariance(hess_inv, variance):
    return hess_inv * variance

# Uncertainty of parameters
def uncertainty(hess_inv, variance):
    cov = covariance(hess_inv, variance)
    return np.sqrt(np.diag(cov))


age, ans = np.loadtxt("/Users/apple/Desktop/survey.csv",delimiter=",",skiprows=1,unpack=True)
x0 = np.array([0.5,0.5])   #initial guess
result = optimize.minimize(neg_log_like, x0=x0, args=(age, ans))   # minimize!!
hess_inv = result.hess_inv    # inverse of hessian matrix
var = result.fun/(len(ans)-len(x0))   # variance
uncer = uncertainty(hess_inv, var)    # uncertainty in beta_0 and beta_1
print('beta_0, beta_1: ', result.x, '\nUncertainty: ', uncer)
print('cov matrix: ', covariance(hess_inv, var))


bins = np.arange(0,85,5)  # size 5; bins of size 10 can be achieved with (0,91,10)
mid_bin = []     # middle point of each bin
bin_height = []
for i in range(16):
    sum = 0     # sum of answers within this bin
    num = 0     # number of answers within this bin
    for j in range(len(age)):
        if (age[j] > bins[i]) and (age[j] <= bins[i+1]):
            sum += ans[j]
            num += 1
    avg = 0 if num == 0 else sum/num   # average answer within this bin (height)
    bin_height.append(avg)
    mid = (bins[i+1] + bins[i])/2    # middle point of bin
    mid_bin.append(mid)
x = np.linspace(0,80,501)
plt.plot(x, prob(x,result.x[0],result.x[1]), "olive")
plt.scatter(age, ans)
plt.scatter(mid_bin, bin_height)
plt.legend(["log model","data","data average"])
plt.xlabel("Age")
plt.ylabel("Likelihood")
plt.title("Logistic Model and Actual Data Points")
plt.show()


