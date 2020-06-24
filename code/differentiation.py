"""
Numerical differentiation
    - dfdx
    - dfdx2
    - dfxydx
    - dfxydy
    - dfxydx2
    - dfxydy2
    - dfxydxy
"""


# %% library loading
import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt


# %% derivatives of an univariate function
def dfdx(f, x, method='central', h=0.001):
    if method == 'forward':
        deriv = (f(x + h) - f(x)) / h
    elif method == 'backward':
        deriv = (f(x) - f(x - h)) / h
    elif method == 'central':
        deriv = (f(x + h) - f(x - h)) / (2 * h)
    else:
        raise ValueError("Method is unknown.")

    return deriv


def dfdx2(f, x, h=0.001):
    deriv = (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)
    return deriv


# %% derivatives of a bivariate function
def dfxydx(f, x, y, h=0.001, k=0.001):
    deriv = (f(x + h, y) - f(x - h, y)) / (2 * h)
    return deriv


def dfxydy(f, x, y, h=0.001, k=0.001):
    deriv = (f(x, y + k) - f(x, y - k)) / (2 * k)
    return deriv


def dfxydx2(f, x, y, h=0.001, k=0.001):
    deriv = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h * h)
    return deriv


def dfxydy2(f, x, y, h=0.001, k=0.001):
    deriv = (f(x, y + k) - 2 * f(x, y) + f(x, y - k)) / (k * k)
    return deriv


def dfxydxy(f, x, y, h=0.001, k=0.001):
    deriv = (f(x + h, y + k) - f(x + h, y - k) - f(x - h, y + k) + f(x - h, y - k)) / (4 * h * k)
    return deriv


# %% several functions to illustrate
def f_1(x): return np.sqrt(1 - x ** 2)


def f_2(x): return x ** sin(x) + x ** cos(x) - np.sqrt(x)


def f_3(x): return sin(x)


def f_4(x): return x ** 2 * sin(x) ** 3


def f_4d1(x): return x * sin(x) ** 2 * (2 * sin(x) + 3 * x * cos(x))


def f_4d2(x): return sin(x) * ((2 - 3 * x * x) * sin(x) ** 2 + 6 * x * x * cos(x) ** 2 + 12 * x * sin(x) * cos(x))


def f_5(x, y): return cos(x ** 4) + 3 * y * y


def f_6(x): return np.sqrt(x) * np.log(x)


def f_6d1(x): return (np.log(x) + 2) / (2 * np.sqrt(x))


def f_6d2(x): return -np.log(x) / (4 * x * np.sqrt(x))


# %% illustration
if __name__ == "__main__":
    plt.figure(figsize=(7, 5))
    x = np.linspace(-6, 6, 200)
    dy = dfdx(f_4, x, method='central', h=1e-5)
    dY = f_4d1(x)
    plt.plot(x, dy, 'r.', label='Central difference 1st derivative')
    plt.plot(x, dY, 'b', label='Exact 1st derivative')
    
    dy = dfdx2(f_4, x, h=1e-5)
    dY = f_4d2(x)
    plt.plot(x, dy, 'm.', label='Central difference 2nd derivative')
    plt.plot(x, dY, 'y', label='Exact 2nd derivative')
    plt.title('Central difference derivatives of y = x^2 * sin(x)^3')
    plt.legend(loc='best')
    plt.show()
