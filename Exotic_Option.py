'''
The function library

Modification log:

7/20/2016 - creation of the file

'''
import pandas as pd
import numpy as np
import scipy.stats

def tildeQPDF(alpha, b, tau, eta = -1):
    '''
    a very general tilde_q PDF function, used for lookback option
    '''
    term1_ = np.exp(-np.power(b-alpha*tau, 2)/2/tau)/np.power(2*np.pi*tau, 0.5)
    term2_ = eta*alpha*np.exp(2*alpha*b) * scipy.stats.norm.cdf(eta*(b + alpha * tau)/np.power(tau,0.5))
    return 2* (term1_ + term2_)


def brownianBridgeUToB(x0, xt, sig, t, max = True):
    '''
    Return the b of a Briwnian Bridge by randomly generate a U(0,1]
    '''

    dx_ = xt - x0

    if max:
        return x0 + 0.5 * (dx_ + np.power(dx_*dx_ \
               - 2*sig**2*t*np.log(np.random.uniform(0,1)), 0.5))
    else:
        return x0 + 0.5 * (dx_ - np.power(dx_*dx_ \
               - 2*sig**2*t*np.log(np.random.uiform(0,1)), 0.5))
