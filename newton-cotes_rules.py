import numpy as np


def newton_cotes(f, a, b, N=50, method="simpson"):
    """
    method
        trapezoid   : Riemann sum using right endpoints
        simpson     : Riemann sum using left endpoints
        simpson38   : Riemann sum using midpoints
    """
    if method == "trapezoid":
        dx  = (b-a)/N
        x   = np.linspace(a,b,N+1)
        y   = f(x)
        val = dx/2 * np.sum(y[0:-1:] + y[1::])

    elif method == "simpson":
        N  += 2 - N % 2
        dx  = (b-a)/N
        x   = np.linspace(a,b,N+1)
        y   = f(x)
        val = dx/3 * np.sum(y[0:-1:2] + 4*y[1::2] + y[2::2])

    elif method == "simpson38":
        N  += 3 - N % 3
        dx  = (b-a)/N
        x   = np.linspace(a,b,N+1)
        y   = f(x)
        val = 3*dx/8 * np.sum(y[0:-1:3] + 3*y[1::3] + 3*y[2::3] + y[3::3])

    elif method == "midpoint":
        N  += 2 - N % 2
        dx  = (b-a)/N
        x   = np.linspace(a,b,N)
        y   = f(x)
        val = 2*dx * np.sum(y[1::2])

    elif method == "trapezoid_open":
        N  += 3 - N % 3
        dx  = (b-a)/N
        x   = np.linspace(a,b,N)
        y   = f(x)
        val = 3*dx/2 * np.sum(y[1:-1:3] + y[2::3])

    elif method == "milne":
        N  += 4 - N % 4
        dx  = (b-a)/N
        x   = np.linspace(a,b,N)
        y   = f(x)
        val = 4*dx/3 * np.sum(2*y[1:-1:4] - y[2::4] + 2*y[3::4])

    print('Newton-Cote {:15} integral = {:20.15f}'.format(method, val))


def f_1(x): return np.sqrt(1-x**2)


def f_2(x): return x ** np.sin(x) + x ** np.cos(x) - np.sqrt(x)


def f_3(x): return np.sin(x)


def f_4(x): return x**2 * np.sin(x)**3


def f_5(x): return 3*x**2


newton_cotes(f_4, a=10, b=20, N=10000, method="trapezoid")
newton_cotes(f_4, a=10, b=20, N=10000, method="simpson")
newton_cotes(f_4, a=10, b=20, N=10000, method="simpson38")
newton_cotes(f_4, a=10, b=20, N=10000, method="midpoint")
newton_cotes(f_4, a=10, b=20, N=10000, method="trapezoid_open")
newton_cotes(f_4, a=10, b=20, N=10000, method="milne")

print("Wolframalpha 'integrate x^2 sin^3 x dx from 10 to 20' = -181.157840953674")