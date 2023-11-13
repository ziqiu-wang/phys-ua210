import numpy as np
from scipy import optimize as opt

def f(x):
    return ((x-0.3)**2)*(np.exp(x))

def quad_interp(a,b,c):
    fa = f(a)
    fb = f(b)
    fc = f(c)
    denom = (b - a) * (fb - fc) - (b - c) * (fb - fa)
    numer = (b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa)
    if (np.abs(denom) < 1.e-15):
        x = b
    else:
        x = b - 0.5 * numer / denom
    return x

def sort_abc(a,b,c):
    # swaps the values of a,b,c so that f(b) < f(a) and f(b) < f(c)
    val = [a,b,c]
    f_val = [f(a),f(b),f(c)]
    min = f_val[0]
    ind_min = 0
    for i in [1,2]:
        if f_val[i] < min:
            min = f_val[i]
            ind_min = i
    b = val[ind_min]   # sets b to the value with smallest f
    val.remove(b)
    a, c = val[0], val[1]   #sets a and c to the other two values
    return a, b, c

def myBrent():
    a = -1.5
    b = 0.5
    c = 5.0
    mach_prec = 10**(-5)   #machine precision
    gold_sec = (3. - np.sqrt(5)) / 2.0
    used_gold = True   # whether golden section is used in the last iteration
    diff = np.abs(max(a,b,c) - min(a,b,c))  # size of bracketing interval
    approx_lst = [b]    # approximation in each iteration

    while np.abs(diff) > mach_prec:
        s = approx_lst[-1]   # b of previous iteration
        r = quad_interp(a,b,c)
        approx_lst.append(r)
        print(r)
        if (((r > max(a,b,c)) or (r < min(a,b,c)))
                or ((used_gold == False) and (len(approx_lst)>=4)
                and (abs(approx_lst[-1] - approx_lst[-2]) >= abs(approx_lst[-3] - approx_lst[-4])))):
            used_gold = True
            # use golden section search
            a, b, c = sort_abc(a, b, c)
            if ((b - a) > (c - b)):
                x = b
                b = b - gold_sec * (b - a)
            else:
                x = b + gold_sec * (c - b)
            fb = f(b)
            fx = f(x)
            if (fb < fx):
                c = x
            else:
                a = b
                b = x
            approx_lst[-1] = b    # change the current guess to golden section guess
            # sort a, b, c simply by their values for upcoming quadratic interpolation
            brac = [a, b, c]
            brac.sort()
            a, b, c = brac[0], brac[1], brac[2]
        else:
            # update golden section bounds
            used_gold = False
            if r < s:
                c = s
            else:
                a = s
            b = r
        diff = max(a,b,c) - min(a,b,c)

    print("Final Answer:", approx_lst[-1])

myBrent()
root = opt.brent(f, brack=(0,5))
print(f"Scipy answer: {root}")






