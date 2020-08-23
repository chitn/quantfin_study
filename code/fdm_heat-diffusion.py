"""
Finite difference method for heat-defusion problem
"""

# %% load necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d


# %% problem definition
length = 1
time   = 2.5
temp   = 100
cond   = 0.01
splt   = 200
method = 'cn'


# %% numerical properties
nx, nt = 100, 10000
dx, dt = length/nx, time/nt
inp    = [cond, nx, dx, nt, dt]
print(dx, dt)


# %% initial condition
"""
# in a triangular form
u0  = np.zeros((nx+1))
mid = int(nx/2)
for i in range(mid+1):
    u0[i]     = temp*i/mid
    u0[i+mid] = temp*(1-i/mid)
"""
u0 = temp*np.sin(np.pi * np.linspace(0,length,nx+1) / length)
u0[0] = u0[-1] = 0


def exact_sol(t):
    return temp*np.sin(np.pi * np.linspace(0,length,nx+1) / length) * np.exp(-cond*np.pi*np.pi*t*dt/(length*length))


def plot_field(inp, u, splt):
    _, nx, dx, nt, dt = inp

    x = np.linspace(0, length, nx + 1)
    error = list()
    error.append(0)

    figure = plt.figure(figsize=(15, 5))
    ax_one = figure.add_subplot(1, 3, 1)
    ax_one.plot(x, u[0,:], 'ro')

    for t in range(1, nt + 1):
        if t % splt == 0:
            ax_one.plot(x, u[t,:], 'b-', linewidth=0.5)
        u_exact = exact_sol(t)
        error.append(np.linalg.norm(u[t,:] - u_exact) / np.sqrt(nx))

    ax_two = figure.add_subplot(1,3,2,projection='3d')
    ax_two.view_init(30,-120)
    X = np.linspace(0,length,nt+1)
    Y = np.linspace(0,time,nx+1)
    X, Y = np.meshgrid(X, Y)
    surf = ax_two.plot_surface(X, Y, u.transpose(),cmap=cm.jet)

    X = np.linspace(0,length,nt+1)
    ax_three = figure.add_subplot(1,3,3)
    ax_three.plot(X,error,'ro')

    plt.show()


# %% solving with FDM Explicit
def explicit(inp, u0):
    print('Explicit scheme...')
    cond, nx, dx, nt, dt = inp

    k = cond*dt/(dx*dx)
    if k >= 0.5:
        print("   Unstable scheme k = {:5.3f}, please change time-/space-step or conductivity.".format((k)))
        # raise ValueError("Unstable scheme k = {:5.3f}, please change time- or space-step or conductivity.".format((k)))
        return

    x = np.linspace(0,length,nx+1)
    u = np.zeros((nt+1,nx+1))

    u[0,:] = u0

    for t in range(1,nt+1):
        u_new = np.zeros((nx+1))
        for i in range(1,nx):
            u_new[i] = u0[i] + k*(u0[i+1] - 2*u0[i] + u0[i-1])
        u_new[0]  = 0
        u_new[nx] = 0
        u0 = u_new
        u[t,:] = u0

    print("   Succesfully execute.")
    return u


# %% solving with FDM Implicit
def implicit(inp, u0):
    print('Implicit scheme...')
    cond, nx, dx, nt, dt = inp

    # forming relationship matrix
    k = cond*dt/(dx*dx)
    A = np.zeros((nx+1,nx+1))
    for i in range(1,nx):
        A[i,i-1] = -k
        A[i,i]   = 1+2*k
        A[i,i+1] = -k
    A[0,0] = A[nx,nx] = 1
    A[0,1] = A[nx,nx-1] = 0
    A = np.linalg.inv(A)

    x = np.linspace(0,length,nx+1)
    u = np.zeros((nt+1,nx+1))

    u[0,:] = u0

    for t in range(1,nt+1):
        u0 = np.dot(A, u0)
        u0[0]  = 0
        u0[nx] = 0
        u[t,:] = u0

    print("   Succesfully execute.")
    return u


# %% solving with FDM Implicit
def crank_nicholson(inp, u0):
    print('Crank-Nicholson scheme...')
    cond, nx, dx, nt, dt = inp

    # forming relationship matrix
    k = cond*dt/(2*dx*dx)
    A = np.zeros((nx+1,nx+1))
    B = np.zeros((nx+1,nx+1))

    A[0,0] = A[nx,nx] = 1
    A[0,1] = A[nx,nx-1] = 0
    B[0,0] = 1 - 2*k
    B[0,1] = k
    B[nx,nx-1] = k
    B[nx,nx] = 1 - 2*k

    for i in range(1,nx):
        A[i,i-1] = -k
        A[i,i]   = 1+2*k
        A[i,i+1] = -k

        B[i,i-1] = k
        B[i,i]   = 1-2*k
        B[i,i+1] = k

    A = np.linalg.inv(A)

    x = np.linspace(0,length,nx+1)
    u = np.zeros((nt+1,nx+1))

    u[0,:] = u0

    for t in range(1,nt+1):
        tmp = np.dot(B, u0)
        tmp[0] = tmp[nx] = 0
        u0  = np.dot(A, tmp)
        u[t,:] = u0

    print("   Succesfully execute.")
    return u


if method == 'explicit':
    u_num = explicit(inp, u0)
elif method == 'implicit':
    u_num = implicit(inp, u0)
elif method == 'cn':
    u_num = crank_nicholson(inp, u0)
else:
    raise ValueError('Unkown method...')
plot_field(inp, u_num, splt)


    
    

