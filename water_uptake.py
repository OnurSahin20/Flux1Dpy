from numba import njit
import numpy as np

def feddes_model(p0, p0opt, p2h, p2l, p3, r2h, r2l, bx):
    @njit(fastmath=True, cache=True)
    def simulator(h, tp):
        size = h.shape[0]
        sink = np.empty(size, dtype=bx.dtype)
        
        if tp >= r2h:
            p2 = p2h
        elif tp <= r2l:
            p2 = p2l
        else:
            p2 = p2l + (p2h - p2l) * (tp - r2l) / (r2h - r2l)
            
        for i in range(size):
            hi = h[i]  # Store locally for slightly faster memory access
            if hi >= p0 or hi <= p3:
                # Out of bounds (anaerobiosis or wilting point)
                alpha = 0.0
                
            elif p0opt <= hi < p0:
                # Linear decrease due to lack of oxygen
                alpha = (hi - p0) / (p0opt - p0)
                
            elif p2 <= hi < p0opt:
                alpha = 1.0
                
            else: 
                alpha = (hi - p3) / (p2 - p3)
            
            sink[i] = alpha * bx[i] * tp
            
        return sink
        
    return simulator
   
def sshape_model(p0, p50, bx):
    inv_p50 = 1.0 / np.abs(p50)
    @njit(fastmath=True, cache=True)
    def simulator(h, tp):
        size = h.shape[0]
        sink = np.empty(size, dtype=bx.dtype) 
        for i in range(size):
            if h[i] >= 0.0:
                sink[i] = 0.0
            else:
                suction_ratio = np.abs(h[i]) * inv_p50
                alpha = 1.0 / (1.0 + suction_ratio ** p0)
                sink[i] = alpha * bx[i] * tp
        return sink 
        
    return simulator