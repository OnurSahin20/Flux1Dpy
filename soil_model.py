from numba import njit
import numpy as np

def vgm_model(tr, ths, ks, a, n, m, L): 
    @njit(cache=True,fastmath=True)
    def simulator(h,sm=False):
        size = h.shape[0]
        theta,conduct,cap = np.empty(size, dtype=tr.dtype), np.empty(size, dtype=tr.dtype),np.empty(size, dtype=tr.dtype)
        if sm:
            if h[i] >= 0.0:
                theta[i] = ths[i]
                conduct[i] = ks[i]
                cap[i] = 1e-6
            else:
                ah = np.abs(a[i] * h[i])
                ah_n = ah ** n[i]
                base = 1.0 + ah_n
                se = base ** -m[i]
                theta[i] = tr[i] + (ths[i] - tr[i]) * se
            
            return theta, conduct, cap
        else:
            for i in range(size):
                if h[i] >= 0.0:
                    theta[i] = ths[i]
                    conduct[i] = ks[i]
                    cap[i] = 1e-6
                else:
                    ah = np.abs(a[i] * h[i])
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
    size = tr.shape[0]
    hs = -2.0
    
    # Pre-calculate the static air-entry arrays ONCE during setup
    ah_s = a * np.abs(hs)
    se_s = (1.0 + ah_s ** n) ** -m
    thm_arr = tr + (ths - tr) / se_s
    se_s_1m = se_s ** (1.0 / m)
    term_den_arr = (se_s ** L) * (1.0 - (1.0 - se_s_1m) ** m) ** 2

    @njit(cache=True,fastmath=True)
    def simulator(h,sm=False):
        theta = np.empty(size, dtype=tr.dtype)
        conduct = np.empty(size, dtype=tr.dtype)
        cap = np.empty(size, dtype=tr.dtype)
        if sm:
            for i in range(size):
                if h[i] >= hs:
                    theta[i] = ths[i]
                    conduct[i] = ks[i]
                    cap[i] = 1e-6
                else:
                    # Only calculate the dynamic h-dependent terms inside the loop
                    ah = np.abs(a[i] * h[i])
                    base = 1.0 + ah ** n[i]
                    se_star = base ** -m[i]
                    thm = thm_arr[i]
                    term_den = term_den_arr[i]
                    theta[i] = tr[i] + (thm - tr[i]) * se_star
            return theta,conduct,cap  

        else:
            for i in range(size):
                if h[i] >= hs:
                    theta[i] = ths[i]
                    conduct[i] = ks[i]
                    cap[i] = 1e-6
                else:
                    # Only calculate the dynamic h-dependent terms inside the loop
                    ah = np.abs(a[i] * h[i])
                    base = 1.0 + ah ** n[i]
                    se_star = base ** -m[i]
                    
                    # Fetch precalculated static terms from the closure
                    thm = thm_arr[i]
                    term_den = term_den_arr[i]
                    
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
    @njit(cache=True,fastmath=True)
    def simulator(h):
        theta = np.empty(size, dtype=tr.dtype)
        conduct = np.empty(size, dtype=tr.dtype)
        cap = np.empty(size, dtype=tr.dtype)
        
        for i in range(size):
            if h[i] >= hb[i]:
                theta[i] = ths[i]
                conduct[i] = ks[i]
                cap[i] = 1e-6
            else:
                ratio = hb[i] / h[i]
                ratio_lamb = ratio ** lamb[i]
                
                theta[i] = tr[i] + (ths[i] - tr[i]) * ratio_lamb
                conduct[i] = ks[i] * ratio ** (2.0 + 3.0 * lamb[i])
                cap[i] = -(lamb[i] * (ths[i] - tr[i]) / h[i]) * ratio_lamb
                
        return theta, conduct, cap
    return simulator

@njit(cache=True)
def generate_lut_arrays(simulator, h_table):
    """
    Fast loop to populate LUT arrays, now including the analytical Capacity table.
    """
    num_materials = h_table.shape[0]
    bins = h_table.shape[1]
    
    theta_table = np.empty((num_materials, bins), dtype=h_table.dtype)
    K_table = np.empty((num_materials, bins), dtype=h_table.dtype)
    
    
    for j in range(bins):
        h_slice = np.ascontiguousarray(h_table[:, j])
        
        # Capture theta, K, AND the analytical capacity (cap)
        th, k, cap = simulator(h_slice, False) 
        
        theta_table[:, j] = th
        K_table[:, j] = k
        
    return theta_table, K_table


def lut_model(h_table, theta_table, K_table,mat_ids):
    
    nodes = len(mat_ids)
    bins = h_table.shape[1]

    @njit(cache=True, fastmath=True)
    def simulator(h, sm=False):
        theta = np.empty(nodes, dtype=h_table.dtype)
        conduct = np.empty(nodes, dtype=h_table.dtype)
        cap = np.empty(nodes, dtype=h_table.dtype)
        
        for i in range(nodes):
            m = mat_ids[i]  # Fetch the specific material ID for this node
            h_val = h[i]
            
            # Find index in monotonically increasing h_table for this material
            idx = np.searchsorted(h_table[m], h_val)
            
            if idx == 0:
                h0, h1 = h_table[m, 0], h_table[m, 1]
                t0, t1 = theta_table[m, 0], theta_table[m, 1]
                
                # Calculate the exact chord slope of the final bin
                slope = (t1 - t0) / (h1 - h0)
                
                # Linearly extrapolate theta beyond the table limit
                theta[i] = t0 + slope * (h_val - h0)
                
                # Capacity is perfectly tied to the extrapolated theta
                cap[i] = slope
                if not sm: conduct[i] = K_table[m, 0]
               
                
            elif idx == bins:
                # Saturated / Ponding conditions (h >= 0.0)
                theta[i] = theta_table[m, -1]
                if not sm: conduct[i] = K_table[m, -1]
                cap[i] = 1e-6
                
            else:
                # Linear Interpolation
                h0, h1 = h_table[m, idx-1], h_table[m, idx]
                t0, t1 = theta_table[m, idx-1], theta_table[m, idx]
                
                weight = (h_val - h0) / (h1 - h0)
                theta[i] = t0 + weight * (t1 - t0)
                
                # Static Capacity C(h) using the chord slope for solver fallback
                cap[i] = (t1 - t0) / (h1 - h0) 
                
                if not sm:
                    k0, k1 = K_table[m, idx-1], K_table[m, idx]
                    conduct[i] = k0 + weight * (k1 - k0)
                else:
                    conduct[i] = 0.0 
                    
        return theta, conduct, cap
        
    return simulator
       
        