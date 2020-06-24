"""
Numerical integration
    - Riemann sum
    - Newton-Cote method
    - Monte-Carlo method
"""


# %% library loading
import numpy as np
import math
import matplotlib.pyplot as plt


# %% Riemann sum
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


# %% Newton-Cotes rules
def newton_cotes(f, a, b, N=50, method="simpson"):
    """
    Newton-Cotes closed
        trapezoid    
        simpson     
        simpson38
    Newton-Cotes opened
        trapezoid_open
        midpoint
        milne
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

    print('Newton-Cote {:15} integral = {:10.5f}'.format(method, val))


# %% Monte-Carlo methods
def monte_carlo_proportion(f, x_range, N, plot_yn):
    # geometrical method
    x0 = min(x_range)
    x1 = max(x_range)
    x = np.arange(x0, x1, 0.01)
    y = f(x)
    z = np.zeros(N)
    f0, f1 = min(min(y), 0), max(y)

    x_rand = x0 + np.random.random(N) * (x1 - x0)
    y_rand = f0 + np.random.random(N) * (f1 - f0)

    count = 0
    for i in range(N):
        yx = y_rand[i]
        fx = f(x_rand[i])
        if math.fabs(yx) <= math.fabs(fx):
            if (yx > 0) and (fx > 0) and (yx <= fx):  # area over x-axis is positive
                count += 1
                z[i] = 1
            if (yx < 0) and (fx < 0) and (yx >= fx):  # area under x-axis is negative
                count -= 1
                z[i] = 2

    print("Monte-Carlo geometrical method : ", (f1 - f0) * (x1 - x0) * count / N)

    if plot_yn:
        ind_out = np.where(z == 0)
        ind_pos = np.where(z == 1)
        ind_neg = np.where(z == 2)

        plt.plot(x, y, color="blue", linewidth=2)
        pts_out = plt.scatter(x_rand[ind_out], y_rand[ind_out], s=1, color="yellow")
        pts_pos = plt.scatter(x_rand[ind_pos], y_rand[ind_pos], s=1, color="green")
        pts_neg = plt.scatter(x_rand[ind_neg], y_rand[ind_neg], s=1, color="red")

        # plt.legend((pts_out, pts_pos, pts_neg),
        #            ('Outside', 'Positive', 'Negative'),loc='lower center',ncol=3,fontsize=12)
        plt.show()


def monte_carlo_intx(f, x_range, N):
    # naieve method for univariate functions
    tmp = 0
    x0, x1 = min(x_range), max(x_range)

    for i in range(N):
        x = np.random.uniform(x0, x1)
        tmp += f(x)
    area = x1 - x0
    print("Monte-Carlo naieve for univariate functions : ", area * tmp / N)


def monte_carlo_intxy(f, x_range, y_range, N):
    # naieve method for bivariate functions
    tmp = 0
    x0, x1 = min(x_range), max(x_range)
    y0, y1 = min(y_range), max(y_range)

    for i in range(N):
        x = np.random.uniform(x0, x1)
        y = np.random.uniform(y0, y1)
        tmp += f(x, y)
    area = (x1 - x0) * (y1 - y0)
    print("Monte-Carlo naieve for bivariate functions  : ", area * tmp / N)
    

# %% several functions to illustrate
def f_1(x): return np.sqrt(1-x**2)


def f_2(x): return x ** np.sin(x) + x ** np.cos(x) - np.sqrt(x)


def f_3(x): return np.sin(x)


def f_4(x): return x**2 * np.sin(x)**3


def f_5(x,y): return np.cos(x**4) + 3*y*y


# %% illustration
if __name__ == "__main__":
    # Riemann sum
    riemann_sum(f=f_4, a=10, b=20, N=50, method='trapezoid', plotyn=True)
    riemann_sum(f=f_4, a=10, b=20, N=50, method='right'    , plotyn=True)
    riemann_sum(f=f_4, a=10, b=20, N=50, method='midpoint' , plotyn=True)
    riemann_sum(f=f_4, a=10, b=20, N=50, method='left'     , plotyn=True)
    
    
    # Newton-Cotes rules
    newton_cotes(f=f_4, a=10, b=20, N=10000, method="trapezoid")
    newton_cotes(f=f_4, a=10, b=20, N=10000, method="simpson")
    newton_cotes(f=f_4, a=10, b=20, N=10000, method="simpson38")
    newton_cotes(f=f_4, a=10, b=20, N=10000, method="trapezoid_open")
    newton_cotes(f=f_4, a=10, b=20, N=10000, method="midpoint")
    newton_cotes(f=f_4, a=10, b=20, N=10000, method="milne")
    print("Wolframalpha 'integrate x^2 sin^3 x dx from 10 to 20' = -181.157840953674")
    
    
    # Monte-Carlo method
    N = 1000000
    x_range = [10, 20]
    monte_carlo_proportion(f=f_4, x_range=x_range, N=N, plot_yn=True)
    monte_carlo_intx(f=f_4, x_range=x_range, N=N)
    # integrate x^sin(x) + x^cos(x) - sqrt(x) for x=0 to 5
    
    monte_carlo_intxy(f = f_5, x_range = [4,6], y_range = [0,1], N = N)
    # integrate cos(x^4) + 3*y^2 dx dy for x=4 to 6 y=0 to 1

