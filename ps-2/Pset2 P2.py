import numpy
import numpy as np
import timeit

#for loop solution
start = timeit.default_timer()

L = 150
V_tot = 0
for i in range(-L, L+1):
    for j in range(-L, L+1):
        for k in range(-L, L+1):
            if (i == 0 and j == 0 and k == 0):
                continue
            elif (i+j+k) % 2 == 0:
                V_tot += 1 / np.sqrt(i ** 2 + j ** 2 + k ** 2)
            else:
                V_tot -= 1 / np.sqrt(i ** 2 + j ** 2 + k ** 2)
print(V_tot)

stop = timeit.default_timer()
print("time of method 1:", stop - start)

#without for loop
start = timeit.default_timer()

number_range = np.arange(-L, L+1, 1)
i, j, k = numpy.meshgrid(number_range, number_range, number_range)
root_sum_of_sqrs = np.sqrt(i ** 2 + j ** 2 + k ** 2).flatten()
part_1 = root_sum_of_sqrs[::2].copy() * (-1)**L #this and the next line splits the array into two to add signs
part_2 = root_sum_of_sqrs[1::2].copy() * (-1)**(L+1)
modified_root = np.append(part_1, part_2) #get the full array
modified_root_final = modified_root[modified_root != 0] #getting rid of the 0 element due to (0,0,0)
inverse = 1 / modified_root_final
V_tot = np.sum(inverse)

print(V_tot)

stop = timeit.default_timer()
print("time of method 2:", stop - start)



