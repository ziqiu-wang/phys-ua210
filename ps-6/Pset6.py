import copy
import scipy.linalg as la
import numpy as np
import astropy.io.fits
import matplotlib.pyplot as plt
import timeit

# Part (a)
hdu_list = astropy.io.fits.open("/Users/apple/Desktop/specgrid.fits")
logwave = hdu_list["LOGWAVE"].data
flux = hdu_list["FLUX"].data
flux_backup = copy.deepcopy(flux)
for i in range(0,5):
    f = flux[i]
    plt.plot(logwave, f)
plt.xlabel("$\log_{10}{\lambda}$ $(\lambda$ in $˚A)$")
plt.ylabel("Flux ($10^{-17} {erg s^{−1} cm^{−2} ˚A^{−1}}$)")
plt.title("Data for Five Galaxies")
plt.show()

# Part (b)
norms = []
for gal in flux:
    sum = np.sum(gal)
    norms.append(sum)
    gal /= sum

# Part (c)
mean_flux = []
res_flux = copy.deepcopy(flux)    #residual matrix
for gal in res_flux:
    mean = np.sum(gal)/len(gal)
    mean_flux.append(mean)
    gal -= mean

# Part (d)
start = timeit.default_timer()
print(len(res_flux), len(res_flux[0]))
res_tp = np.transpose(res_flux)    #transpose of residual matrix
cov = np.matmul(res_tp, res_flux)   #covariance matrix, 4001 by 4001
print("cov: done.")
eigs = la.eig(cov, right=True)
print("eig: done.")
stop = timeit.default_timer()
print("part (d) time:", stop-start)
eigval = eigs[0]         #eigenvalues
eigvec = eigs[1]         #eigenvectors
for i in range(5):
    vec = eigvec[:,i]
    plt.plot(logwave, vec)
plt.xlabel("$\log_{10}{\lambda}$ $(\lambda$ in $˚A)$")
plt.ylabel("Eigenvector Components")
plt.title("First Five Eigenvectors of Covariance Matrix")
plt.show()

# Part (e)
start = timeit.default_timer()
U, W, VT = np.linalg.svd(res_tp, full_matrices=False)
V = np.transpose(VT)       #new eigenvectors using SVD
eigvec2 = np.matmul(res_tp, V)
stop = timeit.default_timer()
print("part (e) time:", stop-start)
print(eigvec.size)
print(eigvec2.size)

eigvec_tp = np.transpose(eigvec2)    #transpose so that eigenvectors are the rows for convenience
eigvec_tp_unnorm = copy.deepcopy(eigvec_tp)
for vec in eigvec_tp:
    norm = -la.norm(vec)     #normalize the eigenvectors from SVD
    vec /= norm

for i in range(5):
    vec = eigvec_tp[i]
    plt.plot(logwave, vec)
plt.xlabel("$\log_{10}{\lambda}$ $(\lambda$ in $˚A)$")
plt.ylabel("Eigenvector Components")
plt.title("First Five Eigenvectors of Covariance Matrix Using SVD")
plt.show()

# Part (f)
cond1 = np.linalg.cond(cov)
cond2 = np.linalg.cond(res_tp)
print("condition number of C:", cond1)
print("condition number of RT:", cond2)

# Part (g)
approx_res = np.matmul(np.transpose(VT[:5]), eigvec_tp_unnorm[:5])  #approximate residuals
approx_spec = copy.deepcopy(approx_res)
for i in range(len(norms)):
    approx_spec[i] += np.array(mean_flux)[i]
    approx_spec[i] *= np.array(norms)[i]               #approximate spectra

for i in range(5):
    plt.plot(logwave, approx_spec[i])
plt.xlabel("$\log_{10}{\lambda}$ $(\lambda$ in $˚A)$")
plt.ylabel("Flux ($10^{-17} {erg s^{−1} cm^{−2} ˚A^{−1}}$)")
plt.title("Approximate Spectra for Five Galaxies")
plt.show()

# Part (h)
c0 = V[:,0]
c1 = V[:,1]
c2 = V[:,2]

plt.scatter(range(len(c0)), c0/c1)
plt.scatter(range(len(c0)), c0/c2)
plt.legend(["$c_{0,i}/c_{1,i}$", "$c_{0,i}/c_{2,i}$"])
plt.xlabel("$i$-th Component of $c$")
plt.ylabel("Ratio")
plt.title("$c_0/c_1$ and $c_0/c_2$")
plt.show()

#Part (i)
tot_sqr_res = []
flux = flux_backup
for i in range(1,21):
    approx_res = np.matmul(np.transpose(VT[:i]), eigvec_tp_unnorm[:i])  #approximate residuals
    approx_spec = copy.deepcopy(approx_res)
    for j in range(len(norms)):
        approx_spec[j] += np.array(mean_flux)[j]
        approx_spec[j] *= np.array(norms)[j]
    approx_spec[approx_spec == 0] += 0.1  # avoids the divide by zero warning
    frac_res = ((approx_spec - flux)/approx_spec)**2
    total_res = np.sum(np.ndarray.flatten(frac_res))   #total squared residuals of all galaxies and wavelengths
    tot_sqr_res.append(total_res)

plt.plot(range(1,21), np.log10(tot_sqr_res))
plt.xlabel("$N_c$")
plt.ylabel("Log of Total Residuals")
plt.title("Total Squared Fractional Residuals")
plt.show()

