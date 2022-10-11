import numpy as np

'''
Centered Moving Average of a column vector x over
window k
'''
def movmean(x,k):
    '''
    Centered Moving Average of a column vector x over
    window k
    '''
    assert isinstance(x, np.ndarray)
    assert isinstance(k, int)
    y = np.zeros(shape=x.shape[0])
    for i in range(x.shape[0]):
        if i < (k)//2:
            y[i] = np.mean(x[:i+(k-1)//2+1])
        else:
            y[i] = np.mean(x[i-k//2:i+(k-1)//2+1])
    return y

def dB_to_linear(x):
    return 10**(x/10)

def dBm_to_watt(x):
    return 10**(x/10)/1000

#to be tested
# def movmean(x, k=3) :
#     ret = np.cumsum(x, dtype=float)
#     ret[k:] = ret[k:] - ret[:-k]
#     return ret[k - 1:] / k