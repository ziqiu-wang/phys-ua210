import numpy as np
import matplotlib.pyplot as plt

#creating all the c values
N = 3500
number_range = np.linspace(-2,2,N)
c_reals = np.repeat(number_range, N)
c_imags = np.tile(number_range, N)
c_values = np.empty(N**2, dtype=np.complex128)
c_values.real = c_reals
c_values.imag = c_imags
print(c_values)

#function that checks whether a number is in the set
def checkMandelbrot(c):
    z = 0
    for i in range(100):
        z = z**2 + c
        if np.abs(z)>2:
            return False
    return True

#vectorizing the function to let it operate on arrays
checkMandelbrotByArray = np.vectorize(checkMandelbrot)

#boolean array that indicates whether a corresponding "c" value is in the set
bool_arr = checkMandelbrotByArray(c_values)

#getting all points in the set by using bool_arr as a mask
allPoints = c_values[bool_arr]

#plotting
realPart = allPoints.real
imagPart = allPoints.imag
plt.scatter(realPart, imagPart, color="black", s=3)
plt.ylabel('Imaginary')
plt.xlabel('Real')
plt.title('Points in the Mandelbrot Set')
plt.show()
