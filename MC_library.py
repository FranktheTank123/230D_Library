'''
The function library

Modification log:

7/20/2016 - add multiStepMC and it's derivatives

6/29/2016 - enhance the functions, more taliored functions

6/25/2016 - creation of the file

'''

import pandas as pd
import numpy as np
import scipy.stats
import time


def sampleSummary(sample):
    '''
    return a dataframe containning the summary, given a np matrix. the summary
    is applied on each cols of the matrix
    '''
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



def BoxMuller(uni_array):
    '''
    Box-Muller implementation, note that if uni_array is odd, only the first n-1 elements will be used
    '''
    n = uni_array.size
    if n%2: ## make sure the size is even
        uni_array = uni_array[:-1]
    normal_array = np.empty(n)
    normal_array[::2] = (-2*np.log(1-uni_array[::2]))**0.5 * np.cos(2*np.pi*uni_array[1::2])
    normal_array[1::2] = (-2*np.log(1-uni_array[::2]))**0.5 * np.sin(2*np.pi*uni_array[1::2])
    return normal_array


def oneStepMC(S, K, T, sig, r, y, z, t=0, anti = False):
    '''
    one step MC
    '''
    if anti:
        z = -np.array(z)
    else:
        z = np.array(z)
    return S * np.exp((r-y - sig**2/2)*(T-t) + sig * (T-t)**0.5 * z) # z could be an array

def multiStepMC(S, K, T, sig, r, y, z, t=0, anti = False
                , tracker = lambda S_ts : S_ts):
    '''
    multi-step-mc:
    ***NOTE THE STEPS IS DETERMINED BY THE DIMENSION OF Z (which is a np.array)***
    if Z is 1d array -> automatic use oneStepMC

    assume equally spaced time steps

    tracker: a function (takes an array of S_ts) that keep track of features of the evolution of the
    stock price, which could be max/min, or whether a boundary is hitted
    '''

    if(len(z.shape) == 1): ## this means Z is 1d array, use oneStepMC
        return oneStepMC(S, K, T, sig, r, y, z, t, anti)
    else:
        if anti:
            z = -z

        steps_ = z.shape[1] ## z is M by N here
        dt_ = (T-t)/steps_

        ## increments_ is also a M by N array
        increments_ =  np.exp((r - y - sig**2/2)*dt_ + sig * dt_**0.5 * z)

        ## sum all the rows, using comprod along the rows
        '''
        now S0 is also included in S_ts
        '''
        S_ts = np.column_stack((np.repeat(S,z.shape[0]), \
                 S * np.cumprod(increments_,axis = 1)))
        #print(S_ts)
        return S_ts[:,-1], np.apply_along_axis(tracker, 1, S_ts)


def multiStepMC_generic(z, price_evolution, anti = False
                , tracker = lambda S_ts : S_ts):
    '''
    multi-step-mc:
    ***NOTE THE STEPS IS DETERMINED BY THE DIMENSION OF Z (which is a np.array)***

    assume equally spaced time steps


    price_evolution: a function that takes an 1d array of Z slice and
                     returns 1d array (+1 size to include s0) of the evlotion
                     of underlyings which based on the Zs

    tracker: a function (takes an array of evolution of underlyings)
            that keep track of features of the evolution of the
            stock price, which could be max/min, or whether a boundary is hitted
    '''

    if anti:
        z = -z

    ## generate the evolution of underlyings for all pathes
    #print(z)
    evolutions_ = np.apply_along_axis(price_evolution, 1, z)

    return evolutions_[:,-1], np.apply_along_axis(tracker, 1, evolutions_)




def getHedgeError(st_s, getDelta, terminalPrice, init_balance, y, r, T):
    '''
    given an evolution of the stock price change (S0 not included),
    this function discretely hedge the delta,
    the final balance (i.e. error) is returned

    Parameters:
    st_s: an evolution of stock price, with S0 included
    getDelta: a function takes s_t and t, and return the corresponding delta
    terminalPrice: option evaluation at T
    init_balance: the premium of selling the option
    r: risk-free interest rate
    T: maturity time
    '''

    #dt_ = T/(len(st_s)-1) ## determine the size of each time steps
    time_steps_ = np.linspace(0,T,len(st_s)) ## all the time steps

    ## this should give st_s the delta of each time steps
    deltas_ = getDelta(st_s, time_steps_)

    ## collecting the dividend pay between t, t+dt, and discount it
    disc_dividend_gain_ = deltas_[:-1]*st_s[:-1]*(np.exp(y*time_steps_[1])-1)\
                        *np.exp(r*(T-time_steps_[1:]))

    ## change of deltas
    change_of_deltas_ = np.append(deltas_[0],np.diff(deltas_))
    rebalance_cost = -st_s*change_of_deltas_ ## net cost on rebalancing
    ## discount the rebalance cost
    disc_rebalance_cost = rebalance_cost * np.exp(r*(T-time_steps_))

    ## final payoff = sum of cashflows + FV of selling the options
    ##      + FV of stock price * unit - option payment + dividend
    error_ = disc_rebalance_cost[:-1].sum() + init_balance*np.exp(r*T)\
            + st_s[-1]*deltas_[-2] - terminalPrice(st_s[-1]) \
            + disc_dividend_gain_.sum()

    return error_


def mmTransform( T, sig, r, y, z, t=0):
    '''
    assume matching exp mean and log var
    '''
    z = np.array(z)
    beta = 1./np.var(z)
    alpha = sig * (T-t)**0.5 / 2. - np.log(np.mean(np.exp(beta * sig * (T-t)**0.5 * z))) / (sig * (T-t) ** 0.5)
    return (alpha + beta * z, alpha, beta)



def getBSVar(S, K, T, sig, r, y, t=0):
    '''
    specific for 3.b.ii), which is the variance of standard BS call
    '''
    _d_1 = (np.log(S/K) + ( r - y + sig **2/2)*(T-t)) / ( sig * (T-t)**0.5)
    _d_2 = _d_1 - (sig * (T-t)**0.5)
    _d_3 = _d_1 + (sig * (T-t)**0.5)

    return ( S**2 * np.exp((2*r-2*y+sig**2)*(T-t)) *  scipy.stats.norm.cdf(_d_3) +
            K**2 *scipy.stats.norm.cdf(_d_2) -
            2*K*S* np.exp((r-y)*(T-t)) * scipy.stats.norm.cdf(_d_1) -
            (bsCall(S, K, T, sig, r, y, t=t) * np.exp(r*(T-t))) ** 2  )


def callCal(S_T, K, T, sig, r,t=0):
    '''
    standard call option evaluation
    '''
    result = S_T-K
    result[result<0] = 0
    return np.exp(-r*(T-t))*result


def exoticCall_analytic(S, K, T, sig, r, y, t=0):
    '''
    taliored function for 4.d
    the analytical solution from HW2
    '''
    _y_eff = r
    _r_eff = 1.5 * r - y / 2 - 3.*sig**2/8
    _sig_eff = sig / 2
    _S_eff = S**0.5
    _K_eff = K**0.5

    _d_1 = (np.log(_S_eff/_K_eff) + ( _r_eff - _y_eff +_sig_eff**2/2)*(T-t)) / (_sig_eff * (T-t)**0.5)
    _d_2 = _d_1 - (_sig_eff * (T-t)**0.5)
    return (np.exp(-_y_eff*(T-t))*scipy.stats.norm.cdf(_d_1) * _S_eff -
            np.exp(-_r_eff*(T-t))*scipy.stats.norm.cdf(_d_2) * _K_eff ) * 2*K / _S_eff



def exoticCal(S_T, K, T, sig, r,t=0):
    '''
    taliored function for 4.a
    '''
    result = (S_T ** 0.5 - K ** 0.5) * 2 * K / S_T**0.5
    result[result<0] = 0
    return np.exp(-r*(T-t))*result



def mcTimer (size, get_c_t ,seed = 1):
    '''
    a wrapper function for 2.c.iii
    '''
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




def bbmcWrapper(S, K, T, sig, r, y):
    '''
    A VERY useful wrapper
    this wrapper will take a size of normal array, and return its c_t array using bbmc
    '''
    return lambda z, S=S: callCal( oneStepMC(S, K, T, sig, r, y, z), K, T, sig, r)


def amcWrapper(S, K, T, sig, r, y):
    '''
    A VERY useful wrapper
    this wrapper will take a size of normal array, and return its c_t array using amc
    '''
    return lambda z, S=S: (callCal(oneStepMC(S, K, T, sig, r, y, z, anti = False), K, T, sig, r) +
                      callCal(oneStepMC(S, K, T, sig, r, y, z, anti = True), K, T, sig, r))/2.


def bbmcWrapper_exotic(S, K, T, sig, r, y):
    '''
    A VERY useful wrapper
    this wrapper will take a size of normal array, and return its c_t array using bbmc
    '''
    return lambda z, S=S: exoticCal(oneStepMC(S, K, T, sig, r, y, z), K, T, sig, r)


def cvmcWrapper_exotic(S, K, T, sig, r, y):
    '''
    A VERY useful wrapper
    this wrapper will take a size of normal array, and return its c_t array using bbmc
    '''
    return lambda z, S=S: (exoticCal(oneStepMC(S, K, T, sig, r, y, z), K, T, sig, r) -
                            callCal( oneStepMC(S, K, T, sig, r, y, z), K, T, sig, r) +
                            bsCall(S, K, T, sig, r, y))



def deltaSensitivity( z_array, s_t, get_c_t, shock, mode = 0 ):
    '''
    mode 0: delta +
    mode 1: delta -
    mode 2: central delta

    z_array:    an array of simulated normal variables
    s_t:        a given stock price at time t
    get_c_t:    will get the call value given s
    shock:      new_s_t = (1+shock)*s_t
    '''
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


def gammaSensitivity( z_array, s_t, get_c_t, shock ):
    '''
    z_array:    an array of simulated normal variables
    s_t:        a given stock price at time t
    get_c_t:    will get the call value given s
    shock:      new_s_t = (1+shock)*s_t

    ## now only support central gamma
    '''
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
    '''
    calculate the analytical BS call value
    '''
    _d_1 = (np.log(S/K) + (r-y+sig**2/2)*(T-t)) / (sig * (T-t)**0.5)
    _d_2 = _d_1 - (sig * (T-t)**0.5)
    return np.exp(-y*(T-t))*scipy.stats.norm.cdf(_d_1) * S - np.exp(-r*(T-t))*scipy.stats.norm.cdf(_d_2) * K


def callDelta(S, K, T, sig, r, y, t=0):
    '''
    calculate the analytical BS call delta value
    '''

    _d_1 = (np.log(S/K) + (r-y+sig**2/2)*(T-t)) / (sig * (T-t)**0.5)
    delta_ =  np.exp(-y*(T-t))*scipy.stats.norm.cdf(_d_1)
    #delta_[np.where( T == t )] = S>=K ## take care of extreme case
    return delta_


def callGamma(S, K, T, sig, r, y, t=0):
    '''
    calculate the analytical BS call gamma value
    '''
    _d_1 = (np.log(S/K) + (r-y+sig**2/2)*(T-t)) / (sig * (T-t)**0.5)
    return np.exp(-y*(T-t))*scipy.stats.norm.pdf(_d_1) / (S * sig * (T-t)**0.5 )

def callVega(S, K, T, sig, r, y, t=0):
    '''
    calculate the analytical BS call vega value
    '''
    _d_1 = (np.log(S/K) + (r-y+sig**2/2)*(T-t)) / (sig * (T-t)**0.5)
    return S*np.exp(-y*(T-t))*scipy.stats.norm.cdf(_d_1)*(T-t)**0.5


def sinsitivity_wrapper(desired_dfs, shocks, S_t_mesh, normal_sample, MCWrapper,
                        K, T, sig, r, y,
                        want_delta_plus = True, want_delta_minus = True, want_delta_c = True,
                        want_delta_real = True, want_gamma_c = True, want_gamma_real = True,
                        true_delta = lambda s_t, K, T, sig, r, y : callDelta(s_t, K, T, sig, r, y),
                        true_gamma = lambda s_t, K, T, sig, r, y : callGamma(s_t, K, T, sig, r, y)):
    '''
    desired_dfs and shocks should have same size
    '''
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




def numerical_delta(price_func, bump = 1e-5):
    '''
    numerical delta calculation (central diff only..)
    imput function is of the for fun(s_t, K, T, sig, r, y)
    '''
    return lambda s_t, K, T, sig, r, y: (price_func( (1+bump)*s_t , K, T, sig, r, y) \
    - price_func( (1-bump)*s_t, K, T, sig, r, y))/(2*bump * s_t)


def numerical_gamma(price_func, bump = 1e-5):
    '''
    numerical gamma calculation (central diff only..)
    imput function is of the for fun(s_t, K, T, sig, r, y)
    '''
    return lambda s_t, K, T, sig, r, y: (price_func((1+bump)*s_t , K, T, sig, r, y) \
    + price_func((1-bump)*s_t , K, T, sig, r, y) - 2 * price_func(s_t, K, T, sig, r, y)) \
    /(bump * s_t)**2
