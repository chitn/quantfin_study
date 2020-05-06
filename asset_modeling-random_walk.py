"""
Created on April 28 2020
The random walk model for return modelling follows Introduces Quantitative Finances of Paul Wilmott.
"""

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


def r6():
    ans = 0
    for i in range(6):
        ans += random.random()
    ans -= 3
    return ans
    
def rand(numb):
    if numb == 6:
        return r6()
    else:
        return r6() + r6()
    
    
for j in range(0,scenario):
    for i in range(1,step):
        asset[i] = asset[i-1]*(1 + dt + vt*rand(6))    
        areturn[i] = (asset[i] - asset[i-1])/asset[i-1]
        
    ax1.plot(range(0,step), asset)
    ax2.hist(areturn, num_bins, normed = True)
    
plt.show()    
