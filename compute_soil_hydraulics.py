# gets the required parameters for solver per iteration. 
import calculateConductivity
import calculateMoisture
from numba import njit
import numpy as np 

@njit("float64(float64[::1],float64[::1],float64[::1],float64[::1],float64[::1],float64[::1],float64[::1],float64[::1])", fastmath=True)
def get_props_vgm(h_vec,a,n,m,L,tr,ths,ks) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    k,th,c = np.empty(h_vec.shape),np.empty(h_vec.shape),np.empty(h_vec.shape)
    for i in range(0,h_vec.shape[0]):
        k[i] = calculateConductivity.get_conduct_vgm(h_vec[i],a[i],n[i],m[i],L[i],ks[i])
        th[i] = calculateMoisture.moisture_vgm(h_vec[i],a[i],n[i],m[i],tr[i],ths[i])
        # analytical formulation of capacity for VGM. It is challenging for 
        c[i] = (a[i] * m[i] * n[i] * (ths[i] - tr[i]) * np.power(np.abs(a[i] * h_vec[i]), n[i] - 1.0)) / np.power(
            1.0 + np.power(np.abs(a[i] * h_vec[i]), n[i]), m[i] + 1.0) if h_vec[i] < 0.0 else 0.0
    return k,th,c 