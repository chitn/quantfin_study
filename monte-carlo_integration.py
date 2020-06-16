import numpy as np
import math
import matplotlib.pyplot as plt


def monte_carlo_proportion(f, x_range, N, plot_yn):
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

    print("Value of the integral : ", (f1 - f0) * (x1 - x0) * count / N)

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
    tmp = 0
    x0, x1 = min(x_range), max(x_range)

    for i in range(N):
        x = np.random.uniform(x0, x1)
        tmp += f(x)
    area = x1 - x0
    print("Value of the integral : ", area * tmp / N)


def monte_carlo_intxy(f, x_range, y_range, N):
    tmp = 0
    x0, x1 = min(x_range), max(x_range)
    y0, y1 = min(y_range), max(y_range)

    for i in range(N):
        x = np.random.uniform(x0, x1)
        y = np.random.uniform(y0, y1)
        tmp += f(x, y)
    area = (x1 - x0) * (y1 - y0)
    print("Value of the integral : ", area * tmp / N)


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


N = 1000000
x_range = [10, 20]
monte_carlo_proportion(f=f_4, x_range=x_range, N=N, plot_yn=False)
monte_carlo_intx(f=f_4, x_range=x_range, N=N)
# integrate x^sin(x) + x^cos(x) - sqrt(x) for x=0 to 5

# monte_carlo_intxy(f = f_3, x_range = [4,6], y_range = [0,1], N = N)
# integrate cos(x^4) + 3*y^2 dx dy for x=4 to 6 y=0 to 1
