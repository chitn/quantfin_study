"""
Root-finding algorithms
    - Bisection method
    - Newton-Raphson method
    - Secant method
"""


# %% library loading
import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt


# %% Bisection method
def bisection(f, a, b, epsilon=1e-10, N_max=100):
    if (f(a) * f(b) > 0):
        raise ValueError("f(a)*f(b) > 0, choose a better value for a or b.")
        return None, None
    if f(a) == 0: return a, 0
    if f(b) == 0: return b, 0

    a_new = a
    b_new = b
    m_new = (a_new + b_new) / 2
    f_new = f(m_new)
    N = 0
    while (abs(f_new) > epsilon) and (N <= N_max):
        if f(a_new) * f_new < 0:
            b_new = m_new
        else:
            a_new = m_new
        m_new = (a_new + b_new) / 2
        f_new = f(m_new)
        N += 1

    if (abs(f_new) < epsilon):
        return m_new, N
    else:
        raise ValueError("Method fails within {%d} iterations.".format(N_max))
        return None, None


# %% Newton-Raphson method for a univariate function
def newton_raphson(f, df, x0, epsilon=1e-10, N_max=1000, opt="numerical"):
    x_new = x0
    f_new = f(x0)
    N = 0
    while (abs(f_new) > epsilon) and (N <= N_max):
        if opt == "analytical": 
            jacobian = df(x_new)
        elif opt == "numerical": 
            jacobian = dfdx(f, x_new)
        else:
            raise ValueError("Method to determine Jacobian is unknown.")
            return None, None
            
        if jacobian == 0:
            raise ValueError("Jacobian vanishes.")
            return None, None
        
        x_new -= f_new / jacobian
        f_new = f(x_new)
        N += 1

    if (abs(f_new) < epsilon):
        return x_new, N
    else:
        raise ValueError("Method fails within {:d} iterations.".format(N_max))
        return None, 


# %% Secant method
def secant(f, a, b, epsilon=1e-10, N_max=100):
    if (f(a) * f(b) > 0):
        raise ValueError("f(a)*f(b) > 0, choose a better value for a or b.")
        return None, None
    if f(a) == 0: return a, 0
    if f(b) == 0: return b, 0

    a_new = a
    b_new = b
    s_new = a_new - f(a_new) * (b_new - a_new) / (f(b_new) - f(a_new))
    f_new = f(s_new)
    N = 0
    while (abs(f_new) > epsilon) and (N <= N_max):
        if f(a_new) * f_new < 0:
            b_new = s_new
        else:
            a_new = s_new
        s_new = a_new - f(a_new) * (b_new - a_new) / (f(b_new) - f(a_new))
        f_new = f(s_new)
        N += 1

    if (abs(f_new) < epsilon):
        return s_new, N
    else:
        raise ValueError("Method fails within {%d} iterations.".format(N_max))
        return None, None


# %% several functions to illustrate
def f_1(x): return np.sqrt(1 - x ** 2)


def f_2(x): return x ** sin(x) + x ** cos(x) - np.sqrt(x)


def f_3(x): return sin(x)


def f_4(x): return x ** 2 * sin(x) ** 3


def f_4d1(x): return x * sin(x) ** 2 * (2 * sin(x) + 3 * x * cos(x))


def f_5(x, y): return cos(x ** 4) + 3 * y * y


def f_6(x): return np.sqrt(x) * np.log(x) - 5


def f_6d1(x): return (np.log(x) + 2) / (2 * np.sqrt(x))


# %% numerical differentiation
def dfdx(f, x, method='central', h=1e-10):
    if method == 'forward':
        deriv = (f(x + h) - f(x)) / h
    elif method == 'backward':
        deriv = (f(x) - f(x - h)) / h
    elif method == 'central':
        deriv = (f(x + h) - f(x - h)) / (2 * h)
    else:
        raise ValueError("Method is unknown.")

    return deriv


# %% illustration
if __name__ == "__main__":
    root, iter = bisection(f=f_6, a=1, b=10, epsilon=1e-10)
    f = abs(f_6(root))
    print("Bisection method           : {:3d} {:10.7f} {:7.5e}".format(iter, root, f))
    
    root, iter = newton_raphson(f=f_6, df=f_6d1, x0=3, epsilon=1e-10, opt="numerical")
    f = abs(f_6(root))
    print("Newton-Raphson numerical   : {:3d} {:10.7f} {:7.5e}".format(iter, root, f))
    
    root, iter = newton_raphson(f=f_6, df=f_6d1, x0=3, epsilon=1e-10, opt="analytical")
    f = abs(f_6(root))
    print("Newton-Raphson analytical  : {:3d} {:10.7f} {:7.5e}".format(iter, root, f))
    
    root, iter = secant(f=f_6, a=1, b=10, epsilon=1e-10)
    f = abs(f_6(root))
    print("Secant method              : {:3d} {:10.7f} {:7.5e}".format(iter, root, f))
