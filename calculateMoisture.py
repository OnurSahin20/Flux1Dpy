import numpy as np 
from numba import njit

# explicitly dividing methods for faster calculation not hashing dictonary each time inputs are floats s
@njit("float64(float64,float64,float64,float64)", fastmath=True)
def correct_func(h,a,n,m) -> float:
    # function for FXW and FXW-M1.
    return np.power(np.log(np.e + np.power(np.abs(a * h), n)), -m)
    

@njit("float64(float64,float64,float64,float64,float64,float64)", fastmath=True)
def moisture_vgm(h,a,n,m,tr,ths) -> float:
    if (h>=0): 
        moist = ths
    else:
        ah = np.abs(a * h)
        moist = (ths - tr) / np.power(1 + np.power(ah, n), m) + tr
    return moist

@njit("float64(float64,float64,float64,float64,float64)", fastmath=True)
def moisture_fxw(h,a,n,m,ths) -> float:
    hr,h0 = -1500.0,-0.63 * np.power(10, 7)
    moist = (1 - np.log(1 + h / hr) / np.log(1 + h0 / hr)) * correct_func(h,a,n,m) * ths
    return moist


@njit("float64(float64,float64,float64,float64,float64)", fastmath=True)
def moisture_fxw_m1(h,a,n,m,ths) -> float:
    hs,hr,h0 = -1.0,-1500.0,-0.63 * np.power(10, 7)
    if (h < hs):
        ch  = correct_func(h,a,n,m) 
        cs =  correct_func(hs,a,n,m) 
        moist = ths * (1 - np.log(1 + (h - hs) / hr) / np.log(1 + (h0 - hs) / hr)) * ch / cs
    else:
        moist = ths
    return moist


