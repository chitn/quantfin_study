import numpy as np
import math
import matplotlib.pyplot as plt


def riemann_sum(f, a, b, N, method='midpoint', plotyn=True):
    """
    method
        right       : Riemann sum using right endpoints
        left        : Riemann sum using left endpoints
        midpoint    : Riemann sum using midpoints
        trapezoid   : trapezoidal sum
    """
    dx = (b - a)/N
    x = np.linspace(a,b,N+1)

    if plotyn:
        y = f(x)
        # for plotting
        X = np.linspace(a, b, 10 * N + 1)
        Y = f(X)
        plt.figure(figsize=(6,3))
        plt.plot(X,Y,'b')

    if method == 'left':
        x_l = x[:-1]

        if plotyn:
            y_l = y[:-1]
            plt.plot(x_l, y_l, 'b.', markersize=10)
            plt.bar(x_l, y_l, width=dx, alpha=0.2, align='edge', edgecolor='b')
            plt.title('Left Riemann sum, N = {}'.format(N))
            plt.show()

        print('Left Riemann sum, N = {}'.format(np.sum(f(x_l)*dx)))

    elif method == 'right':
        x_r = x[1:]

        if plotyn:
            y_r = y[1:]
            plt.plot(x_r, y_r, 'b.', markersize=10)
            plt.bar(x_r, y_r, width=-dx, alpha=0.2, align='edge', edgecolor='b')
            plt.title('Right Riemann sum, N = {}'.format(N))
            plt.show()

        print('Right Riemann sum, N = {}'.format(np.sum(f(x_r)*dx)))

    elif method == 'midpoint':
        x_m = (x[:-1] + x[1:])/2

        if plotyn:
            y_m = f(x_m)
            plt.plot(x_m, y_m, 'b.', markersize=10)
            plt.bar(x_m, y_m, width=dx, alpha=0.2, edgecolor='b')
            plt.title('Midpoint Riemann sum, N = {}'.format(N))
            plt.show()

        print('Midpoint Riemann sum, N = {}'.format(np.sum(f(x_m)*dx)))

    elif method == 'trapezoid':
        x_l = x[:-1]
        x_r = x[1:]

        if plotyn:
            for i in range(N):
                xs = [x[i], x[i], x[i + 1], x[i + 1]]
                ys = [0, f(x[i]), f(x[i + 1]), 0]
                plt.fill(xs, ys, 'b', edgecolor='b', alpha=0.2)
            plt.title('Trapezoidal sum, N = {}'.format(N))
            plt.show()

        print('Trapezoidal sum, N = {}'.format(dx/2 * np.sum(f(x_l) + f(x_r))))

    else:
        raise ValueError("Method must be 'left', 'right', 'midpoint' or 'trapezoid'.")


def f_1(x):
    return np.sqrt(1-x**2)


def f_2(x):
    return x ** np.sin(x) + x ** np.cos(x) - np.sqrt(x)


def f_3(x):
    return np.sin(x)


def f_4(x):
    return x**2 * np.sin(x)**3


def f_5(x,y):
    return np.cos(x**4) + 3*y*y


riemann_sum(f=f_4, a=10, b=20, N=1000, method='trapezoid', plotyn=True)
# riemann_sum(f=f_2, a=0, b=5, N=30, method='right', plotyn=True)
# riemann_sum(f=f_2, a=0, b=5, N=30, method='midpoint', plotyn=True)
# riemann_sum(f=f_2, a=0, b=5, N=30, method='trapezoid', plotyn=True)

