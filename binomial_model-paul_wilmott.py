"""
Created on May 5 2020
The binomial model for European- and American-style exercises follows Introduces Quantitative Finances of Paul Wilmott.
"""

import math


def payoff_call(asset_price, strike_value):
    if asset_price > strike_value:
        return asset_price - strike_value
    else:
        return 0


def payoff_put(asset_price, strike_value):
    if strike_value > asset_price:
        return strike_value - asset_price
    else:
        return 0
    
    
def printout(arr):
    for row in arr:
        for item in row: print("{:8.4f}".format(item), end = " ")
        print("")
    print("")


asset        = 100
strike       = 110
intrate      = 0.06
volatility   = 0.3
expiry       = 0.25

rec_step     = [2, 4, 8, 16, 32, 64, 128, 256, 512]  
rec_step     = list(range(1,11)) + list(range(20,110,10)) + list(range(150,550,50)) + list(range(600,1100,100))
rec_price_eu = []
rec_price_am = []

for nostep in rec_step:
    dtime      = expiry/nostep
    discount   = math.exp(-intrate * dtime)  
    
    
    tmp = 0.5 * (discount + math.exp((intrate + volatility*volatility)*dtime))
    u   = tmp + math.sqrt(tmp*tmp - 1)
    v   = 1/u
    p   = (math.exp(intrate * dtime) - v)/(u - v)
    S   = [[0 for i in range(nostep + 1)] for j in range(nostep + 1)]
    Veu = [[0 for i in range(nostep + 1)] for j in range(nostep + 1)]
    Vam = [[0 for i in range(nostep + 1)] for j in range(nostep + 1)]


    # Asset value
    S[0][0] = asset
    for i in range(1,nostep+1):
        S[0][i] = S[0][i-1] * v
        for j in range(i,0,-1):
            S[j][i] = S[j-1][i-1] * u
            
            
    # Option value        
    for i in range(nostep+1):
        Veu[i][nostep] =  payoff_put(S[i][nostep] , strike)
        Vam[i][nostep] =  payoff_put(S[i][nostep] , strike)
    
    
    #############################
    #                           #
    #  European-style exercise  #
    #                           #
    #############################
    for i in range(nostep-1,-1,-1):
        for j in range(i+1):
            Veu[j][i] = (p * Veu[j+1][i+1] + (1-p) * Veu[j][i+1]) * discount
    
    
    #############################
    #                           #
    #  American-style exercise  #
    #                           #
    #############################
    for i in range(nostep-1,-1,-1):
        for j in range(i+1):
            payoff_check = payoff_put(S[j][i], strike)
            option_value = (p * Vam[j+1][i+1] + (1-p) * Vam[j][i+1]) * discount
            Vam[j][i]    = max(option_value, payoff_check)
    
            
    price_eu = Veu[0][0]
    price_am = Vam[0][0]
    print("{:5d}  {:8.4f}  {:8.4f}".format(nostep,price_eu,price_am))
    rec_price_eu.append(price_eu)
    rec_price_am.append(price_am)
    
# printout(S)
# printout(Veu)
# printout(Vam)


import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plt.semilogx(rec_step,rec_price_eu,'b--o',label="European-put")  
plt.semilogx(rec_step,rec_price_am,'r-.s',label="American-put") 
plt.xlabel('number of steps')
plt.ylabel('the present option value')  
plt.legend(loc="upper right")
