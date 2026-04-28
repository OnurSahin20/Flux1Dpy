from numba import njit
import numpy as np

def vgm_model(tr, ths, ks, a, n, m, L): 
    @njit(fastmath=True, cache=True)
    def simulator(h):
        size = h.shape[0]
        theta,conduct,cap = np.empty(size, dtype=tr.dtype), np.empty(size, dtype=tr.dtype),np.empty(size, dtype=tr.dtype)
        for i in range(size):
            if h[i] >= 0.0:
                theta[i] = ths[i]
                conduct[i] = ks[i]
                cap[i] = 1e-3
            else:
                ah = a[i] * -h[i]
                ah_n = ah ** n[i]
                base = 1.0 + ah_n
                se = base ** -m[i]
                
                theta[i] = tr[i] + (ths[i] - tr[i]) * se
                
                se_1m = se ** (1.0 / m[i])
                conduct[i] = ks[i] * (se ** L[i]) * (1.0 - (1.0 - se_1m) ** m[i]) ** 2
                
                num = a[i] * m[i] * n[i] * (ths[i] - tr[i]) * (ah ** (n[i] - 1.0))
                den = base ** (m[i] + 1.0)
                cap[i] = num / den
                
        return theta, conduct, cap
    return simulator


def vgm_ae_model(tr, ths, ks, a, n, m, L):
    @njit(fastmath=True, cache=True)
    def simulator(h):
        size = h.shape[0]
        hs = -2.0
        theta,conduct,cap = np.empty(size, dtype=tr.dtype), np.empty(size, dtype=tr.dtype),np.empty(size, dtype=tr.dtype)
        for i in range(size):
            if h[i] >= hs:
                theta[i] = ths[i]
                conduct[i] = ks[i]
                cap[i] = 1e-3
            else:
                ah_s = a[i] * np.abs(hs)
                se_s = (1.0 + ah_s ** n[i]) ** -m[i]
                thm = tr[i] + (ths[i] - tr[i]) / se_s
                term_den = (se_s ** L[i]) * (1.0 - (1.0 - se_s ** (1.0 / m[i])) ** m[i]) ** 2
                ah = a[i] * -h[i]
                ah_n = ah ** n[i]
                base = 1.0 + ah_n
                se_star = base ** -m[i]
                
                theta[i] = tr[i] + (thm - tr[i]) * se_star  
                
                se_star_1m = se_star ** (1.0 / m[i])
                term_num = (se_star ** L[i]) * (1.0 - (1.0 - se_star_1m) ** m[i]) ** 2
                conduct[i] = ks[i] * (term_num / term_den)
                
                num = a[i] * m[i] * n[i] * (thm - tr[i]) * (ah ** (n[i] - 1.0))
                den = base ** (m[i] + 1.0)
                cap[i] = num / den    
        return theta, conduct, cap
    return simulator


def bc_model(hb, ths, tr, lamb, ks):
    size = tr.shape[0]
    @njit(fastmath=True, cache=True)
    def simulator(h):
        theta = np.empty(size, dtype=tr.dtype)
        conduct = np.empty(size, dtype=tr.dtype)
        cap = np.empty(size, dtype=tr.dtype)
        
        for i in range(size):
            if h[i] >= hb[i]:
                theta[i] = ths[i]
                conduct[i] = ks[i]
                cap[i] = 1e-5
            else:
                ratio = hb[i] / h[i]
                ratio_lamb = ratio ** lamb[i]
                
                theta[i] = tr[i] + (ths[i] - tr[i]) * ratio_lamb
                conduct[i] = ks[i] * ratio ** (2.0 + 3.0 * lamb[i])
                cap[i] = -(lamb[i] * (ths[i] - tr[i]) / h[i]) * ratio_lamb
                
        return theta, conduct, cap
    return simulator


       
        