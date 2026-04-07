
from numba.experimental import jitclass
from numba import float64,int32
import numpy as np 
from soil_model import SoilModels
from source_sink import RootWaterUptake
from numba import njit

soil_model_type = SoilModels.class_type.instance_type
root_model_type = RootWaterUptake.class_type.instance_type
spec = {"soil_model":soil_model_type,"root_model":root_model_type,'top_bound':int32,'bot_bound':int32,'z':float64[:],'n':int32,"ha":float64,'hs':float64,
        "A":float64[:],"B":float64[:],"C":float64[:],"F":float64[:], 'head':float64[:], 'pond_max':float64, 'dz_top':float64,'dz':float64[:],
        's1':float64[:],'s2':float64[:],'k':float64[:],'cap':float64[:],'n_1':int32,'n_2':int32} # defining the tridiagonals

@jitclass(spec)
class CreateTriDiagonal:
    def __init__(self,soil_model,root_model,z,top_bound,bot_bound,pond_max):
        self.soil_model = soil_model
        self.root_model = root_model
        self.z = z
        
        self.n = self.z.shape[0]
        self.ha,self.hs = -50000, pond_max
        self.pond_max = pond_max
        self.A,self.B,self.C,self.F = np.zeros(self.n-1),np.zeros(self.n),np.zeros(self.n-1),np.zeros(self.n)
        self.s1,self.s2,self.k,self.cap = np.zeros(self.n),np.zeros(self.n),np.zeros(self.n),np.zeros(self.n)
        self.top_bound,self.bot_bound  = top_bound,bot_bound
        self.n_1,self.n_2 = self.n - 1, self.n - 2
        self.dz_top = abs(self.z[self.n_1] - self.z[self.n_2])
        self.head = np.zeros(self.n)
        self.dz = np.zeros(self.n)
        #self.dz = self.calculate_dz()
    def set_top(self, dt, flux_top, head_top,sink,atmosp):
        if self.top_bound == 0 or self.top_bound == 2: 
            self.B[self.n_1] = 1.0; self.A[self.n_2] = 0.0; self.F[self.n_1] = 0.0;self.head[self.n_1] = head_top
     
        elif atmosp == 0:
            self.B[self.n_1] = 1.0; self.A[self.n_2] = 0.0; self.F[self.n_1] = 0

        else:
            dz_low = self.z[self.n_1] - self.z[self.n_2]
            dz_cell = dz_low / 2.0
            k11 = (self.k[self.n_1] + self.k[self.n_2]) / 2.0
            self.A[self.n_2] = -k11 / (dz_cell * dz_low)
            self.B[self.n_1] = self.cap[self.n_1] / dt + k11 / (dz_cell * dz_low)
            flux_low = k11 * (self.head[self.n_1] - self.head[self.n_2]) / dz_low
            gravity = -k11 / dz_cell 
            self.F[self.n_1] = ((self.s1[self.n_1] - self.s2[self.n_1]) / dt - flux_low / dz_cell + gravity - flux_top / dz_cell - sink[self.n_1])
    
    def set_bot(self, dt, flux_bot, head_bot, sink):
        is_dirichlet = False
        active_head = 0.0
        active_flux = flux_bot

        if (self.bot_bound == 0) or (self.bot_bound == 2): # constant or variable Dirichlet
            is_dirichlet = True
            active_head = head_bot
            
        elif (self.bot_bound == 5):
            dz_bot = self.z[1] - self.z[0] 
            if self.head[0] >= 0.0:
                if self.head[1] > -dz_bot:
                    is_dirichlet = True
                    active_head = 0.0
                else:
                    is_dirichlet = False
                    active_flux = 0.0
            else:
                is_dirichlet = False
                active_flux = 0.0

        if is_dirichlet:
            self.B[0] = 1.0; self.C[0] = 0.0;self.F[0] = 0.0
            self.head[0] = active_head
            
        elif self.bot_bound == 4:
            self.B[0] = 1.0;  self.C[0] = -1.0
            self.F[0] = self.head[1] - self.head[0]
            
        else:
            dz_up = self.z[1] - self.z[0]     
            dz_cell = dz_up / 2.0             
            k01 = (self.k[0] + self.k[1]) / 2.0
            self.C[0] = -k01 / (dz_cell * dz_up)
            self.B[0] = self.cap[0] / dt + k01 / (dz_cell * dz_up)
            flux_up = k01 * (self.head[1] - self.head[0]) / dz_up
            gravity = k01 / dz_cell 
            self.F[0] = ((self.s1[0] - self.s2[0]) / dt + flux_up / dz_cell + gravity + active_flux / dz_cell  - sink[0])
        
    def get_new(self,dt,h1,h2,flux_top,flux_bot,head_top,head_bot,tp,atmosp):
        self.head[:] = h2[:]
        self.root_model.calculate_sink_source(self.head,tp)
        sink = self.root_model.sink  
        self.soil_model.calculate_props(h1,self.head)
        self.s1,self.s2,self.k,self.cap = self.soil_model.theta1,self.soil_model.theta2,self.soil_model.conduct,self.soil_model.capacity
        self.set_top(dt,flux_top,head_top,sink,atmosp)      
        self.set_bot(dt,flux_bot,head_bot,sink)
       
        for i in range(1,self.n-1):
            dz_low = self.z[i] - self.z[i - 1]
            dz_up = self.z[i + 1] - self.z[i]
            dz_cell = (dz_up + dz_low) / 2.0
            if atmosp == 1:
                k1 = (self.k[i] + self.k[i - 1]) / 2.0
                k2 = (self.k[i] + self.k[i + 1]) / 2.0
            else:
                k1 = np.sqrt(self.k[i] * self.k[i - 1])
                k2 = np.sqrt(self.k[i] * self.k[i +1])
            self.A[i - 1] = -k1 / (dz_cell * dz_low)
            self.C[i] = -k2 / (dz_cell * dz_up)
            self.B[i] = (k1 / (dz_cell * dz_low)) + (k2 / (dz_cell * dz_up)) + (self.cap[i] / dt)
            flux_upper = k2 * (self.head[i + 1] - self.head[i]) / dz_up
            flux_lower = k1 * (self.head[i] - self.head[i - 1]) / dz_low
            gravity = (k2 - k1) / dz_cell
            self.F[i] = (self.s1[i] - self.s2[i]) / dt + (flux_upper - flux_lower) / dz_cell  + gravity   - sink[i]
        
        return self.head + solve_thomas(self.A,self.B,self.C,self.F)
        
    def calculate_dz(self):
        self.dz = np.zeros(self.n_1)
        for i in range(self.n_1):
            self.dz[i] = self.z[i+1] - self.z[i]
    
    def calculate_vroot(self, sink):
        return np.sum(sink * self.dz)
@njit
def solve_thomas(A,B,C,F):
    n = B.shape[0]
    alfa,beta,y,h_new = np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n)
    alfa[0] = B[0]
    beta[0] = (C[0] / alfa[0])
    y[0] = F[0] / alfa[0]
    for i in range(1,n):
        alfa[i] = B[i] - A[i - 1] * beta[i - 1]
        if i < n - 1:
            beta[i] = C[i] / alfa[i]
            
        y[i] = (F[i] - y[i - 1] * A[i - 1]) / alfa[i]
        
    h_new[n - 1] = y[n - 1]
    for j in range(1,n):
        r = int(n) - 1 - j
        h_new[r] = y[r] - beta[r] * h_new[r + 1]
    return h_new