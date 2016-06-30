'''
The function library

Modification log:

6/29/2016 - enhance the functions, more taliored functions

6/25/2016 - creation of the file

'''

import pandas as pd
import numpy as np
import scipy.stats
import time


'''
return a dataframe containning the summary, given a np matrix. the summary
is applied on each cols of the matrix
'''
def sampleSummary(sample):
    result_ = pd.DataFrame()
    result_['sample_size'] = [x.size for x in sample]
    result_['mean'] = [np.mean(x) for x in sample]
    result_['variance'] = [np.var(x) for x in sample]
    result_['SD'] = [np.std(x) for x in sample]
    result_['3rd_central_moment'] = [scipy.stats.moment(x, moment=3) for x in sample]
    result_['skewness'] = [scipy.stats.moment(x-np.mean(x), moment=3)/(np.std(x)**3) for x in sample]
    result_['4th_central_moment'] = [scipy.stats.moment(x, moment=4, axis=0) for x in sample]
    result_['kurtosis'] = [scipy.stats.moment(x-np.mean(x), moment=4)/(np.std(x)**4) -3 for x in sample]
    result_['Max'] = [np.max(x) for x in sample]
    result_['Min'] = [np.min(x) for x in sample]
    return result_


'''
Box-Muller implementation, note that if uni_array is odd, only the first n-1 elements will be used
'''
def BoxMuller(uni_array):
    n = uni_array.size
    if n%2: ## make sure the size is even
        uni_array = uni_array[:-1]
    normal_array = np.empty(n)
    normal_array[::2] = (-2*np.log(1-uni_array[::2]))**0.5 * np.cos(2*np.pi*uni_array[1::2])
    normal_array[1::2] = (-2*np.log(1-uni_array[::2]))**0.5 * np.sin(2*np.pi*uni_array[1::2])
    return normal_array



'''
one step MC
'''
def oneStepMC(S, K, T, sig, r, y, z, t=0, anti = False):
    if anti:
        z = -np.array(z)
    else:
        z = np.array(z)
    return S * np.exp((r-y - sig**2/2)*(T-t) + sig * (T-t)**0.5 * z) # z could be an array

# assume matching exp mean and log var
def mmTransform( T, sig, r, y, z, t=0):
    z = np.array(z)
    beta = 1./np.var(z)
    alpha = sig * (T-t)**0.5 / 2. - np.log(np.mean(np.exp(beta * sig * (T-t)**0.5 * z))) / (sig * (T-t) ** 0.5)
    return (alpha + beta * z, alpha, beta)


'''
specific for 3.b.ii), which is the variance of standard BS call
'''
def getBSVar(S, K, T, sig, r, y, t=0):
    _d_1 = (np.log(S/K) + ( r - y + sig **2/2)*(T-t)) / ( sig * (T-t)**0.5)
    _d_2 = _d_1 - (sig * (T-t)**0.5)
    _d_3 = _d_1 + (sig * (T-t)**0.5)

    return ( S**2 * np.exp((2*r-2*y+sig**2)*(T-t)) *  scipy.stats.norm.cdf(_d_3) +
            K**2 *scipy.stats.norm.cdf(_d_2) -
            2*K*S* np.exp((r-y)*(T-t)) * scipy.stats.norm.cdf(_d_1) -
            (bsCall(S, K, T, sig, r, y, t=t) * np.exp(r*(T-t))) ** 2  )

'''
standard call option evaluation
'''
def callCal(S_T, K, T, sig, r,t=0):
    result = S_T-K
    result[result<0] = 0
    return np.exp(-r*(T-t))*result

'''
taliored function for 4.d
the analytical solution from HW2
'''
def exoticCall_analytic(S, K, T, sig, r, y, t=0):
    _y_eff = r
    _r_eff = 1.5 * r - y / 2 - 3.*sig**2/8
    _sig_eff = sig / 2
    _S_eff = S**0.5
    _K_eff = K**0.5

    _d_1 = (np.log(_S_eff/_K_eff) + ( _r_eff - _y_eff +_sig_eff**2/2)*(T-t)) / (_sig_eff * (T-t)**0.5)
    _d_2 = _d_1 - (_sig_eff * (T-t)**0.5)
    return (np.exp(-_y_eff*(T-t))*scipy.stats.norm.cdf(_d_1) * _S_eff -
            np.exp(-_r_eff*(T-t))*scipy.stats.norm.cdf(_d_2) * _K_eff ) * 2*K / _S_eff


'''
taliored function for 4.a
'''
def exoticCal(S_T, K, T, sig, r,t=0):
    result = (S_T ** 0.5 - K ** 0.5) * 2 * K / S_T**0.5
    result[result<0] = 0
    return np.exp(-r*(T-t))*result


'''
a wrapper function for 2.c.iii
'''

def mcTimer (size, get_c_t ,seed = 1):
    _start = time.time()
    np.random.seed(seed)
    _uni_array = np.random.uniform(0,1, size)
    _normal_array = BoxMuller(_uni_array)
    _c_t_array = get_c_t(_normal_array)
    _c_t_estimator = _c_t_array.mean()
    _c_t_var = _c_t_array.var()
    _c_t_SE = scipy.stats.sem(_c_t_array)
    _end = time.time()
    ## return = (time, mean, SE, var)
    #return (_end - _start, _c_t_estimator, _c_t_SE, _c_t_var)
    return (_end - _start,  _c_t_var)



'''
A VERY useful wrapper
this wrapper will take a size of normal array, and return its c_t array using bbmc
'''
def bbmcWrapper(S, K, T, sig, r, y):
    return lambda z, S=S: callCal( oneStepMC(S, K, T, sig, r, y, z), K, T, sig, r)

'''
A VERY useful wrapper
this wrapper will take a size of normal array, and return its c_t array using amc
'''
def amcWrapper(S, K, T, sig, r, y):
    return lambda z, S=S: (callCal(oneStepMC(S, K, T, sig, r, y, z, anti = False), K, T, sig, r) +
                      callCal(oneStepMC(S, K, T, sig, r, y, z, anti = True), K, T, sig, r))/2.

'''
A VERY useful wrapper
this wrapper will take a size of normal array, and return its c_t array using bbmc
'''
def bbmcWrapper_exotic(S, K, T, sig, r, y):
    return lambda z, S=S: exoticCal(oneStepMC(S, K, T, sig, r, y, z), K, T, sig, r)

def cvmcWrapper_exotic(S, K, T, sig, r, y):
    return lambda z, S=S: (exoticCal(oneStepMC(S, K, T, sig, r, y, z), K, T, sig, r) -
                            callCal( oneStepMC(S, K, T, sig, r, y, z), K, T, sig, r) +
                            bsCall(S, K, T, sig, r, y))


'''
mode 0: delta +
mode 1: delta -
mode 2: central delta

z_array:    an array of simulated normal variables
s_t:        a given stock price at time t
get_c_t:    will get the call value given s
shock:      new_s_t = (1+shock)*s_t
'''
def deltaSensitivity( z_array, s_t, get_c_t, shock, mode = 0 ):
    if np.abs(shock)>=1:
        raise Exception('shock is too big!')
    if shock == 0:
        return 0
    if (mode!=0) & (mode!=1)& (mode!=2):
        raise Exception('make sure your model is 0, 1, or 2!')


    _z_array = np.array(z_array)
    _c_0 = get_c_t(_z_array, s_t).mean()
    _c_plus = get_c_t(_z_array, (1+shock)*s_t).mean()
    _c_minus = get_c_t(_z_array, (1-shock)*s_t).mean()


    if mode == 0:
        return (_c_plus - _c_0 ) / (shock * s_t)
    elif mode == 1:
        return (_c_0 - _c_minus ) / (shock * s_t)
    else: ## mode == 2
        return (_c_plus - _c_minus) / (2*shock * s_t)

'''
z_array:    an array of simulated normal variables
s_t:        a given stock price at time t
get_c_t:    will get the call value given s
shock:      new_s_t = (1+shock)*s_t

## now only support central gamma
'''
def gammaSensitivity( z_array, s_t, get_c_t, shock ):
    if np.abs(shock)>=1:
        raise Exception('shock is too big!')

    if shock == 0:
        return 0

    _z_array = np.array(z_array)
    _c_0 = get_c_t(_z_array, s_t).mean()
    _c_plus = get_c_t(_z_array, (1+shock)*s_t).mean()
    _c_minus = get_c_t(_z_array, (1-shock)*s_t).mean()

    return (_c_plus+_c_minus-2*_c_0) / np.power( shock * s_t, 2)


def bsCall(S, K, T, sig, r, y, t=0):
    _d_1 = (np.log(S/K) + (r-y+sig**2/2)*(T-t)) / (sig * (T-t)**0.5)
    _d_2 = _d_1 - (sig * (T-t)**0.5)
    return np.exp(-y*(T-t))*scipy.stats.norm.cdf(_d_1) * S - np.exp(-r*(T-t))*scipy.stats.norm.cdf(_d_2) * K

def callDelta(S, K, T, sig, r, y, t=0):
    _d_1 = (np.log(S/K) + (r-y+sig**2/2)*(T-t)) / (sig * (T-t)**0.5)
    return np.exp(-y*(T-t))*scipy.stats.norm.cdf(_d_1)

def callGamma(S, K, T, sig, r, y, t=0):
    _d_1 = (np.log(S/K) + (r-y+sig**2/2)*(T-t)) / (sig * (T-t)**0.5)
    return np.exp(-y*(T-t))*scipy.stats.norm.pdf(_d_1) / (S * sig * (T-t)**0.5 )


'''
desired_dfs and shocks should have same size
'''
def sinsitivity_wrapper(desired_dfs, shocks, S_t_mesh, normal_sample, MCWrapper,
                        K, T, sig, r, y,
                        want_delta_plus = True, want_delta_minus = True, want_delta_c = True,
                        want_delta_real = True, want_gamma_c = True, want_gamma_real = True,
                        true_delta = lambda s_t, K, T, sig, r, y : callDelta(s_t, K, T, sig, r, y),
                        true_gamma = lambda s_t, K, T, sig, r, y : callGamma(s_t, K, T, sig, r, y)):
    if (len(desired_dfs)!= len(shocks)):
        raise Exception('desired_df and shocks does not have the same size')

    for i in range(len(shocks)):
        # deltas
        if want_delta_plus:
            desired_dfs[i]['delta_+'] = [deltaSensitivity( normal_sample, s_t,
                                              MCWrapper(s_t, K, T, sig, r, y), shocks[i], mode = 0 ) for s_t in S_t_mesh ]
        if want_delta_minus:
            desired_dfs[i]['delta_-'] = [deltaSensitivity( normal_sample, s_t,
                                              MCWrapper(s_t, K, T, sig, r, y), shocks[i], mode = 1 ) for s_t in S_t_mesh ]
        if want_delta_c:
            desired_dfs[i]['delta_c'] = [deltaSensitivity( normal_sample, s_t,
                                              MCWrapper(s_t, K, T, sig, r, y), shocks[i], mode = 2 ) for s_t in S_t_mesh ]
        if want_delta_real:
            desired_dfs[i]['delta_real'] = [true_delta( np.array([s_t]), K, T, sig, r, y)[0] for s_t in S_t_mesh]
        # gammas
        if want_gamma_c:
            desired_dfs[i]['gamma_c'] = [gammaSensitivity( normal_sample, s_t,
                                              MCWrapper(s_t, K, T, sig, r, y), shocks[i]) for s_t in S_t_mesh ]
        if want_gamma_real:
            desired_dfs[i]['gamma_real'] = [true_gamma(np.array([s_t]), K, T, sig, r, y)[0] for s_t in S_t_mesh]

    return desired_dfs



'''
numerical delta calculation (central diff only..)
imput function is of the for fun(s_t, K, T, sig, r, y)
'''
def numerical_delta(price_func, bump = 1e-5):
    return lambda s_t, K, T, sig, r, y: (price_func( (1+bump)*s_t , K, T, sig, r, y) - price_func( (1-bump)*s_t, K, T, sig, r, y))/(2*bump * s_t)



'''
numerical gamma calculation (central diff only..)
imput function is of the for fun(s_t, K, T, sig, r, y)
'''
def numerical_gamma(price_func, bump = 1e-5):
    return lambda s_t, K, T, sig, r, y: (price_func((1+bump)*s_t , K, T, sig, r, y) + price_func((1-bump)*s_t , K, T, sig, r, y) - 2 * price_func(s_t, K, T, sig, r, y))/(bump * s_t)**2
