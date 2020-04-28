# This code reproduces the random walk model presented in Section 4.8 in Paul Wilmott's book
# "Introduces to Quantitative finance, 2nd edition".

import random
import matplotlib.pyplot as plt

asset_init = 100        # initial asset value
drift      = 0.10
volatility = 0.25
timestep   = 1/252      # daily return
duration   = 2          # ten-year duration
scenario   = 500
step       = int(duration/timestep)

asset      = [0]*step
areturn    = [0]*step
asset[0]   = asset_init
dt         = drift*timestep
vt         = volatility*(timestep**0.5)

fig  = plt.figure(figsize=(16,6))
ax1  = fig.add_subplot(121)
ax2  = fig.add_subplot(122)
ax1.set(xlabel = 'time (day)', ylabel = 'close price ($)')
ax2.set(xlabel = 'asset return', ylabel = 'probability')

num_bins   = 20

def rand12():
    rand = random.random() + random.random() + random.random() + \
           random.random() + random.random() + random.random() + \
           random.random() + random.random() + random.random() + \
           random.random() + random.random() + random.random() - 6
    return rand

def rand6():
    rand = random.random() + random.random() + random.random() + \
           random.random() + random.random() + random.random() - 3
    return rand
    
for j in range(0,scenario):
    for i in range(1,step):
        rand = rand12
        asset[i] = asset[i-1]*(1 + dt + vt*rand12())    
        areturn[i] = (asset[i] - asset[i-1])/asset[i-1]
        
    ax1.plot(range(0,step), asset)
    ax2.hist(areturn, num_bins, normed = True)
    
plt.show()    