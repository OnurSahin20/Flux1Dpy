import numpy as np 
from numba.typed import Dict
from numba import njit


@njit("float64(int64, float64, DictType(unicode_type, float64[:]))", fastmath=True)
def correct_func(i,h,soil_param) -> np.ndarray:
    # function for FXW and FXW-M1.
    a,n,m  = soil_param['a'][i],soil_param['n'][i],soil_param['m'][i]
    return np.power(np.log(np.e + np.power(np.abs(a * h), n)), -m)
    

@njit("float64(unicode_type, int64, float64, DictType(unicode_type, float64[:]))", fastmath=True)
def get_moisture(method:str, i:int,h: float, params: Dict) -> float:
    # function calculates the soil moisture retention curve for numeric solver!
    if (method == "VGM"):
        a,n,m,tr,ths = params['a'][i],params['n'][i],params['m'][i],params['tr'][i],params['ths'][i]
        if (h>=0): 
            moist = ths
        else:
            ah = np.abs(a * h)
            moist = (ths - tr) / np.power(1 + np.power(ah, n), m) + tr
    elif (method == 'FXW'):
        hr,h0 = -1500.0,-0.63 * np.power(10, 7)
        ths = params['ths'][i]
        moist = (1 - np.log(1 + h / hr) / np.log(1 + h0 / hr)) * correct_func(i,h,params) * ths

    elif (method == 'FXW-M1'):
        hs,hr,h0 = -1.0,-1500.0,-0.63 * np.power(10, 7)
        ths = params['ths'][i]
        if (h < hs):
            moist = ths * (1 - np.log(1 + (h - hs) / hr) / np.log(1 + (h0 - hs) / hr)
                ) * correct_func(i,h,params) / correct_func(i,hs,params) 
        else:
            moist = ths
    return moist



@njit("float64(unicode_type, int64, float64, DictType(unicode_type, float64[:]))", fastmath=True)
def get_conduct(method:str, i:int,h: float, params: Dict) -> float:
    if (method == 'VGM'):
        a,n,m,L,ks = params['a'][i],params['n'][i],params['m'][i],params['L'][i],params['ks'][i]
        if (h>=0): 
            k = ks
        else:
            se = np.power((1 + np.power(np.abs(a * h), n)), -m)
            k = ks * np.power(se, L) * np.power((1 - np.power((1 - np.power(se, (1 / m))), m)), 2)
        
    elif (method == 'FXW'):
        a,n,m,L,ks = params['a'][i],params['n'][i],params['m'][i],params['L'][i],params['ks'][i]
        h0 = -0.63 * np.power(10, 7)
        rh = correct_func(i,h,params)
        r0 = correct_func(i,h0,params)
        sek = (rh-r0) / (1-r0)
        k = ks * np.power(sek,L) * np.power(1 - np.power(1 - np.power(rh,1/m),1-1/n),2)

    elif (method == 'FXW-M1'):
        a,n,m,L,ks = params['a'][i],params['n'][i],params['m'][i],params['L'][i],params['ks'][i]
        hs,h0 = -1.0,-0.63 * np.power(10, 7)
        if (h < hs):
            rh = correct_func(i,h,params)
            r0 = correct_func(i,h0,params)
            rs = correct_func(i,hs,params)
            nom = 1 - np.power(1 - np.power(rh, 1 / m), 1 - 1 / n)
            denom = 1 - np.power(1 - np.power(rs, 1 / m), 1 - 1 / n)
            k = ks * np.power((rh - r0) / (rs - r0),L) * np.power(nom / denom, 2)
        else:
            k = ks

    return k