import copy
import numpy as np
import matplotlib.pyplot as plt

# Getting data
t, y = np.loadtxt("/Users/apple/Desktop/signal.dat",delimiter="|",skiprows=1,usecols=(1,2),unpack=True)

# Part (a)
plt.scatter(t, y, s=3)
plt.xlabel("time")    # not sure about the units
plt.ylabel("signal")
plt.title("Signal vs. Time")
plt.show()

# Part (b)
# Pseudo-inverse
def pinv(A):
    U, W, VT = np.linalg.svd(A, full_matrices=False)  #SVD
    Winv = copy.deepcopy(W)
    Winv[Winv!=0] = 1/(Winv[Winv!=0])   #Pseudo-inverse of diagonal matrix W
    Ainv = np.matmul(np.matmul(VT.transpose(), np.diag(Winv)), U.transpose())
    return Ainv

A = np.c_[t**0, t**1, t**2, t**3]
pinv_A = pinv(A)   #pseudo-inverse of A
coeff = np.matmul(pinv_A, y)   #coefficients of 3rd order polynomial
print(coeff)

fit_y = np.matmul(A, coeff)   #signal predicted by model

plt.scatter(t, y, s=3)
plt.scatter(t, fit_y, s=3)
plt.xlabel("time")    #not sure about the units
plt.ylabel("signal")
plt.title("Signal vs. Time")
plt.show()

# Part (c)
num = len(t)      #number of data points
res = y - fit_y   #residuals
print(res.sum())
std = 2.0         #standard deviation
p_1std = ((abs(res)<=std).sum())/num   #percentage of points that lie within 1 std
p_2std = ((abs(res)<=(2*std)).sum())/num   #percentage of points that lie within 2 std
print(p_1std)
print(p_2std)

plt.scatter(t, y, s=3)
plt.scatter(t, fit_y, s=3)
plt.scatter(t, res, s=3, c="red")
plt.legend(["actual signal", "curve fit", "residual"])
plt.xlabel("time")
plt.ylabel("signal")
plt.title("Signal vs. Time")
plt.show()

# Part (d)
A = np.c_[t**0, t**1, t**2, t**3, t**4,
          t**5, t**6, t**7, t**8]
pinv_A = pinv(A)
coeff = np.matmul(pinv_A, y)
print(coeff)
fit_y = np.matmul(A, coeff)

plt.scatter(t, y, s=3)
plt.scatter(t, fit_y, s=3)
plt.xlabel("time")
plt.ylabel("signal")
plt.title("Signal vs. Time")
plt.show()

res = y - fit_y   #residuals
print(res.sum())
std = 2.0         #standard deviation
p_1std = ((abs(res)<=std).sum())/num   #percentage of points that lie within 1 std
p_2std = ((abs(res)<=(2*std)).sum())/num   #percentage of points that lie within 2 std

print(p_1std)
print(p_2std)

# Part (e)
longest_period = (t.max()-t.min())/2
smallest_freq = 2*np.pi/longest_period
freq = np.linspace(smallest_freq, smallest_freq*10, 23)
A = np.c_[np.cos(freq[0]*t), np.sin(freq[0]*t),
          np.cos(freq[1]*t), np.sin(freq[1]*t),
          np.cos(freq[2]*t), np.sin(freq[2]*t),
          np.cos(freq[3]*t), np.sin(freq[3]*t),
          np.cos(freq[4]*t), np.sin(freq[4]*t),
          np.cos(freq[5]*t), np.sin(freq[5]*t),
          np.cos(freq[6]*t), np.sin(freq[6]*t),
          np.cos(freq[7]*t), np.sin(freq[7]*t),
          np.cos(freq[8]*t), np.sin(freq[8]*t),
          np.cos(freq[9]*t), np.sin(freq[9]*t),
          np.cos(freq[10]*t), np.sin(freq[10]*t),
          np.cos(freq[11]*t), np.sin(freq[11]*t),
          np.cos(freq[12]*t), np.sin(freq[12]*t),
          np.cos(freq[13]*t), np.sin(freq[13]*t),
          np.cos(freq[14]*t), np.sin(freq[14]*t),
          np.cos(freq[15]*t), np.sin(freq[15]*t),
          np.cos(freq[16]*t), np.sin(freq[16]*t),
          np.cos(freq[17]*t), np.sin(freq[17]*t),
          np.cos(freq[18]*t), np.sin(freq[18]*t),
          np.cos(freq[19]*t), np.sin(freq[19]*t),
          np.cos(freq[20]*t), np.sin(freq[20]*t),
          np.cos(freq[21]*t), np.sin(freq[21]*t),
          np.cos(freq[22]*t), np.sin(freq[22]*t)]
pinv_A = pinv(A)
coeff = np.matmul(pinv_A, y)
print(coeff)
fit_y = np.matmul(A, coeff)

plt.scatter(t, y, s=3)
plt.scatter(t, fit_y, s=3)
plt.xlabel("time")
plt.ylabel("signal")
plt.title("Signal vs. Time")
plt.show()

res = y - fit_y   #residuals
std = 2.0         #standard deviation
p_1std = ((abs(res)<=std).sum())/num   #percentage of points that lie within 1 std
p_2std = ((abs(res)<=(2*std)).sum())/num   #percentage of points that lie within 2 std
print(p_1std)
print(p_2std)

