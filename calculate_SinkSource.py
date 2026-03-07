import numpy as np 
from numba import njit

# source sink calculation s = alpha * b(x) * tp 

@njit("float64(float64,float64,float64,float64,float64,float64,float64,float64,float64)", fastmath=True)
def get_alpha_feddes(h, tp,p0, p0opt, p2h, p2l, p3, r2h, r2l) -> float:
    if tp > r2h:
        p2 = p2h
    elif tp < r2l:
        p2 = p2l
    else:
        p2 = p2l + (p2h - p2l) * (tp - r2l) / (r2h - r2l)

    if h >= p0 or h <= p3:
        alfa = 0
    elif p0opt <= h < p0:
        alfa = (h - p0) / (p0opt - p0)
    elif p0opt > h and h >= p2:
        alfa = 1
    elif p3 <= h < p2:
        alfa = (h - p3) / (p2 - p3)
    else:
        alfa = 0
    return alfa

@njit("float64(float64,float64,float64)", fastmath=True)
def get_alpha_sshape(h,p1,p2) -> float:
    # it calculates alpha for root water uptake sink source term
    return 1.0 / (1.0 + np.power(np.abs(h) / np.abs(p1), p2))


@njit("float64(float64,float64,float64,float64,float64,float64,float64,float64,float64,float64)", fastmath=True)
def get_sink_source_feddes(bx,h, tp,p0, p0opt, p2h, p2l, p3, r2h, r2l) -> float:
    return get_alpha_feddes(h, tp,p0, p0opt, p2h, p2l, p3, r2h, r2l) * bx * tp 
  
@njit("float64(float64,float64,float64,float64,float64)", fastmath=True)
def get_sink_source_sshape(bx,tp,h, p1,p2) -> float:
    return get_alpha_sshape(h,p1,p2) * bx * tp