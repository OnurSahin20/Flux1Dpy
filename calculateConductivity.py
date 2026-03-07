import numpy as np 
from numba import njit

# numba methods for calculation of the conductivity for different soil models.
# explicitly dividing methods for faster calculation not hashing dictonary each time inputs are floats s
@njit("float64(float64,float64,float64,float64)", fastmath=True)
def correct_func(h,a,n,m) -> float:
    # function for FXW and FXW-M1.
    return np.power(np.log(np.e + np.power(np.abs(a * h), n)), -m)

#only this method duplicated. 

@njit("float64(float64,float64,float64,float64,float64,float64)", fastmath=True)
def get_conduct_vgm(h,a,n,m,L,ks) -> float: 
    if (h>=0): 
        k = ks
    else:
        se = np.power((1 + np.power(np.abs(a * h), n)), -m)
        k = ks * np.power(se, L) * np.power((1 - np.power((1 - np.power(se, (1 / m))), m)), 2)
    return k



@njit("float64(float64,float64,float64,float64,float64,float64)", fastmath=True)
def get_conduct_fxw(h,a,n,m,L,ks) -> float: 
    h0,rh,r0 = -0.63 * np.power(10, 7),correct_func(h,a,n,m),correct_func(h0,a,n,m)
    sek = (rh-r0) / (1-r0)
    k = ks * np.power(sek,L) * np.power(1 - np.power(1 - np.power(rh,1/m),1-1/n),2)
    return k

@njit("float64(float64,float64,float64,float64,float64,float64)", fastmath=True)
def get_conduct_fxw(h,a,n,m,L,ks) -> float: 
    hs,h0 = -1.0,-0.63 * np.power(10, 7)
    if (h < hs):
        rh,r0,rs = correct_func(h,a,n,m),correct_func(h0,a,n,m),correct_func(hs,a,n,m)
        nom = 1 - np.power(1 - np.power(rh, 1 / m), 1 - 1 / n)
        denom = 1 - np.power(1 - np.power(rs, 1 / m), 1 - 1 / n)
        k = ks * np.power((rh - r0) / (rs - r0),L) * np.power(nom / denom, 2)
    else:
        k = ks
    return k 