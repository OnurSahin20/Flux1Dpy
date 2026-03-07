import numpy as np 
from numba import njit



@njit("float64(float64[::1],float64[::1],float64[::1],float64[::1])", fastmath=True)
def create_diagonal(a,b,c,f) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    n = b.shape[0]
    alfa,y,beta,x = np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n)
 
    alfa[0] = b[0]
    beta[0] = (c[0] / alfa[0])
    y[0] = f[0] / alfa[0]
    for i in range(1,n):
        alfa[i] = b[i] - a[i - 1] * beta[i - 1]
        beta[i] = c[i] / alfa[i]
        y[i] = (f[i] - y[i - 1] * a[i - 1]) / alfa[i]
    
    x[n - 1] = y[n - 1]
    for  j in range(1,n):
        r = int(n) - 1 - j
        x[r] = y[r] - beta[r] * x[r + 1]
    
    return x # new h vector for iteration. 
