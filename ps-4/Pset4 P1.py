def f(x):
    return x**4 - 2*x + 1

def trapezoid(N, a, b):
    h = (b-a)/N
    s = 0.5 * f(a) + 0.5 * f(b)
    for i in range(1,N):
        s += f(a+i*h)
    return h*s

a = 0.0
b = 2.0
exact = 4.4

I1 = trapezoid(10,a,b)
I2 = trapezoid(20,a,b)
err_estimate = abs(1/3 * (I2-I1))
err_to_exact = abs(I2 - exact)

print(I1)
print(I2)
print(err_estimate)
print(err_to_exact)
print(abs(err_estimate - err_to_exact))