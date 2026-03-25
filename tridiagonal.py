
from numba.experimental import jitclass
from numba import float64,int32
import numpy as np 
from soil_model import SoilModels
from source_sink import RootWaterUptake
from numba import njit


soil_model_type = SoilModels.class_type.instance_type
root_model_type = RootWaterUptake.class_type.instance_type
spec = {"soil_model":soil_model_type,"root_model":root_model_type,'top_boundary':int32,'bot_boundary':int32,'dz':int32,'N':int32,"ha":float64,'hs':float64,
        "A":float64[:],"B":float64[:],"C":float64[:],"F":float64[:]} # defining the tridiagonals

@njit
def solve_thomas(A,B,C,F):
    n = B.shape[0]
    alfa,beta,y,h_new = np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n)
    alfa[0] = B[0]
    beta[0] = (C[0] / alfa[0])
    y[0] = F[0] / alfa[0]
    for i in range(1,n):
        alfa[i] = B[i] - A[i - 1] * beta[i - 1]
        beta[i] = C[i] / alfa[i]
        y[i] = (F[i] - y[i - 1] * A[i - 1]) / alfa[i]
    h_new[n - 1] = y[n - 1]
    for j in range(1,n):
        r = int(n) - 1 - j
        h_new[r] = y[r] - beta[r] *  h_new[r + 1]
    return h_new


@jitclass(spec)
class CreateTriDiagonal:
    def __init__(self,soil_model,root_model,dz,top_bound,bot_bound):
        self.soil_model = soil_model
        self.root_model = root_model
        self.dz = dz
        self.dz2 = np.power(self.dz, 2)
        self.ha,self.hs = -100_000, 0
        self.A,self.B,self.C,self.F = np.zeros(self.N-1),np.zeros(self.N),np.zeros(self.N-1),np.zeros(self.N)
        self.s1,self.s2,self.k,self.cap = np.zeros(self.N),np.zeros(self.N),np.zeros(self.N),np.zeros(self.N)
        self.head = np.zeros(self.N)
        self.top_bound,self.bot_bound  = top_bound,bot_bound
        self.n = self.s1.shape[0]
        self.n_1,self.n_2 = self.n - 1, self.n - 2
    
    def set_top(self,dt,flux_top,head_top,pond,sink):
       
        if self.top_bound == 4: # atmospheric boundary
            if (self.head[self.n_1] < self.ha) and (flux_top>=0): # lower boundary
                self.B[self.n_1] =1; self.A[self.n_1] = 0; self.F[self.n_1] = 0
                self.head[self.n_1] = self.ha

            if (head_top[self.n_1]>=0): # upper boundary handling surface layer if ponding_max > 0 also
                self.B[self.n_1] =1; self.A[self.n_1] = 0; self.F[self.n_1] = 0
                self.head[self.n_1] = pond

        else:
            if (self.top_bound  == 0) or (self.top_bound == 2): # constant or variable head boundary (main function should change time series of heads)
                self.B[self.n_1] =1; self.A[self.n_1] = 0; self.F[self.n_1] = 0
                self.head[self.n_1] = head_top
            else: # constant or variable flux boundary condition!
                k11 = (self.k[self.n_1] + self.k[self.n_2]) / 2
                self.A[self.n_2] = -k11 / self.dz2
                self.B[self.n_1] =  self.cap[self.n_1] / dt + k11 / self.dz2
                self.F[self.n_1] = self.head[self.n_2] * k11 / self.dz2 - self.head[self.n_1] *  k11 / self.dz2 - k11/self.dz+ (
                    self.s1[self.n_1]-self.s2[self.n_1]) / dt - flux_top / self.dz - sink[self.n_1]
    
    def set_bot(self,dt,flux_bot,head_bot,sink):
        if (self.bot_bound == 0) or (self.bot_bound == 2): #constant head
            self.B[0] =1; self.C[0] = 0; self.F[0] = 0
            self.head[0] = head_bot
        elif (self.bot_bound == 1) or (self.bot_bound == 3): # neumann type flux condition 
            k01 = (self.k[1] + self.k[0]) / 2
            self.C[0] = -k01 / self.dz2
            self.B[0] =  self.cap[0] / dt + k01 / self.dz2
            self.F[0] = self.head[1] * k01 / self.dz2 - self.head[0] * k01 / self.dz2 + k01 / self.dz + (
                    self.s1[0] - self.s2[0]) / dt + flux_bot / self.dz - sink[0]
    
        elif (self.bot_bound == 4): # free drainage
            self.B[0] = 1.0; self.C[0] = -1.0; self.F[0] = self.head[1] - self.head[0]

        elif (self.bot_bound ==5): # seepage face
            if self.head[0] >= 0.0:
                self.B[0] = 1.0;  self.C[0] = 0.0
                self.F[0] = -self.head[0] # setting delta_h = -h_old ensures h_new becomes 0.0.
            else:
                k01 = (self.k[0] + self.k[1]) / 2.0
                self.C[0] = -k01 / self.dz2
                self.B[0] = self.cap[0] / dt + k01 / self.dz2
                self.F[0] = (self.head[1] * k01 / self.dz2 - self.head[0] * k01 / self.dz2 + k01 / self.dz 
                            + (self.s1[0] - self.s2[0]) / dt - sink[0]) # it has the flux term as zero
        
    def get_new(self,dt,h1,h2,tp,pond,head_top,head_bot,flux_top,flux_bot):
        self.head[:] = h2[:]
        self.root_model.calculate_sink_source(self.head,tp)
        sink = self.root_model.sink  
        dz2 = np.power(self.dz, 2) 
        self.set_top(dt,flux_top,head_top,pond,sink)
        self.set_bot(dt,flux_bot,head_bot,sink)
        self.soil_model.calculate_props(h1,self.head)
        self.s1,self.s2,self.k,self.cap = self.soil_model.theta1,self.soil_model.theta2,self.soil_model.conduct,self.soil_model.capacity
        
        for i in range(1,self.n-1):
            k1 = (self.k[i] + self.k[i - 1]) / 2
            k2 = (self.k[i] + self.k[i + 1]) / 2
            self.A[i-1] = -k1 / dz2
            self.B[i] = ((k1 + k2) / dz2 + self.cap[i] / dt)
            self.C[i] = -k2 / dz2
            self.F[i] = (self.s1[i] - self.s2[i]) / dt + k2 * (self.head[i + 1] - self.head[i]) / dz2 - k1 * (
                        self.head[i] - self.head[i - 1]) / dz2+ (k2 - k1) / self.dz - sink[i]
        
        return solve_thomas(self.A,self.B,self.C,self.F) + self.head
        