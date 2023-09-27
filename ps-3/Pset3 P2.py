import numpy as np
import matplotlib.pyplot as plt
import timeit

#main function that does the multiplication using for-loops
def matrixMultiply(N, A, B):
    C = np.zeros([N,N], float)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i,j] += A[i,k] * B[k,j]
    return C

#function that creates two N by N arrays
def createArrays(N):
    A = np.zeros([N,N], float) + 2
    B = np.zeros([N,N], float) + 3
    return A, B

#returns the time of computation for given size N
def computeTime(N):
    A, B = createArrays(N)
    start = timeit.default_timer()
    C = matrixMultiply(N, A, B)
    stop = timeit.default_timer()
    time = stop - start
    print(N, ":", time)
    return time

#getting computation time for N = 10, 30, 50...
N = 10
lst_time1 = []
lst_N = []
for i in range(15):
    lst_time1.append(computeTime(N))
    lst_N.append(N)
    N += 20
print(lst_time1)

##############################################################

#function that computes the multiplication using dot()
def matrixDot(A, B):
    C = np.dot(A, B)
    return C

#returns computation time of dot()
def computeTimeDot(N):
    A, B = createArrays(N)
    start = timeit.default_timer()
    C = matrixDot(A, B)
    stop = timeit.default_timer()
    time = stop - start
    print(N, ":", time)
    return time

#getting computation time of dot()
lst_time2 = []
for n in lst_N:
    lst_time2.append(computeTimeDot(n))
print(lst_time2)

#################################################

#plotting computation time against N^3 for both methods
N_cubed = np.array(lst_N) ** 3
times1 = np.array(lst_time1)
times2 = np.array(lst_time2)
plt.plot(N_cubed, times1, "violet")
plt.plot(N_cubed, times2, "blue")
plt.legend(["for-loop", "dot"])
plt.ylabel('computation time (s)')
plt.xlabel('$N^3$')
plt.title('Computation Time vs. $N^3$ for Both Methods')
plt.show()


