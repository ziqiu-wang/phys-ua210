import numpy as np

# Part (a)
def findRoots1(a, b ,c):
    root1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    root2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    return (root1, root2)

example1 = findRoots1(0.001, 1000, 0.001)
print(example1)

#Part (b)
def findRoots2(a, b, c):
    root1 = (2 * c)/(-b - np.sqrt(b ** 2 - 4 * a * c))
    root2 = (2 * c)/(-b + np.sqrt(b ** 2 - 4 * a * c))
    return (root1, root2)

example2 = findRoots2(0.001, 1000, 0.001)
print(example2)

#Part (c)
def findRootsAccurate(a, b, c):
    # We choose the expressions that can avoid subtractive cancellation errors
    if b >= 0:
        root1 = (2 * c)/(-b - np.sqrt(b ** 2 - 4 * a * c))
        root2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    else:
        root1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        root2 = (2 * c)/(-b + np.sqrt(b ** 2 - 4 * a * c))
    return (root1, root2)

exampleAccurate = findRootsAccurate(0.001, 1000, 0.001)
print(exampleAccurate)