from numba import njit
import numpy as np

def Tridiagonal(soil_model, root_model, bot_bound, z,precision):
    size = z.shape[0]
    n_1, n_2 = size - 1, size - 2
    A = np.empty(size - 1, dtype=precision)
    B = np.empty(size, dtype=precision)
    C = np.empty(size - 1, dtype=precision)
    F = np.empty(size, dtype=precision)
    
    alfa = np.empty(size, dtype=precision)
    beta = np.empty(size, dtype=precision)
    y = np.empty(size, dtype=precision)
    h_new = np.empty(size, dtype=precision)

    @njit(fastmath=True, cache=True)
    def update_solution(dt, s1, h2, flux_top, tp, state):
        sink = root_model(h2, tp)
        s2, k, cap = soil_model(h2)
        # Bottom Boundary 
        if bot_bound == 0:  # Free drainage (unit gradient)
            A[0], B[0], C[0], F[0] = 0.0, 1.0, -1.0, h2[1] - h2[0]
        else:               # Groundwater level (Fixed head = 0)
            A[0], B[0], C[0], F[0] = 0.0, 1.0, 0.0,  
            
        if state == 0:  
            B[n_1] = 1.0
            A[n_2] = 0.0
            F[n_1] = 0.0
        else:           
            # Flux type boundary (Neumann)
            dz_low = z[n_1] - z[n_2]
            dz_cell = dz_low / 2.0
            k11 = (k[n_1] + k[n_2]) / 2.0
            A[n_2] = -k11 / (dz_cell * dz_low)
            B[n_1] = cap[n_1] / dt + k11 / (dz_cell * dz_low)
            flux_low = k11 * (h2[n_1] - h2[n_2]) / dz_low
            gravity = -k11 / dz_cell 
            F[n_1] = ((s1[n_1] - s2[n_1]) / dt - flux_low / dz_cell + gravity - flux_top / dz_cell - sink[n_1])
    
        for i in range(1, n_1):
            dz_low = z[i] - z[i - 1]
            dz_up = z[i + 1] - z[i]
            dz_cell = (dz_up + dz_low) / 2.0
            k1 = (k[i] + k[i - 1]) / 2.0
            k2 = (k[i] + k[i + 1]) / 2.0
            A[i - 1] = -k1 / (dz_cell * dz_low)
            C[i] = -k2 / (dz_cell * dz_up)
            B[i] = (k1 / (dz_cell * dz_low)) + (k2 / (dz_cell * dz_up)) + (cap[i] / dt)
            
            flux_upper = k2 * (h2[i + 1] - h2[i]) / dz_up
            flux_lower = k1 * (h2[i] - h2[i - 1]) / dz_low
            gravity = (k2 - k1) / dz_cell
            
            F[i] = (s1[i] - s2[i]) / dt + (flux_upper - flux_lower) / dz_cell + gravity - sink[i]
            
        # Inline Thomas Algorithm 
        n = size
        alfa[0] = B[0]
        beta[0] = C[0] / alfa[0]
        y[0] = F[0] / alfa[0]
        for i in range(1, n):
            alfa[i] = B[i] - A[i - 1] * beta[i - 1]
            if i < n - 1:
                beta[i] = C[i] / alfa[i]
            y[i] = (F[i] - y[i - 1] * A[i - 1]) / alfa[i]
        h_new[n - 1] = y[n - 1]
        for j in range(1, n):
            r = int(n) - 1 - j
            h_new[r] = y[r] - beta[r] * h_new[r + 1]
        
        return h2 + h_new
    return update_solution