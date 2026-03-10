
from numba.experimental import jitclass
from numba import float64,int32
import numpy as np 
from soil_model import SoilModels
from source_sink import RootWaterUptake
from numba import njit


soil_model_type = SoilModels.class_type.instance_type
root_model_type = RootWaterUptake.class_type.instance_type
spec = {"soil_model":soil_model_type,"root_model":root_model_type,'dz':int32,'N':int32,"ha":float64,'hs':float64,'pond_max':float64,
        "A":float64[:],"B":float64[:],"C":float64[:],"F":float64[:]} # defining the tridiagonals

@njit
def solve_thomas(A,B,C,F):
    n = B.shape[0]
    alfa,beta,y,h_new = np.zeros(n),np.zeros(n),np.zeros(beta),np.zeros(n)
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
    def __init__(self,soil_model,dz):
        self.soil_model = soil_model
        self.N = self.soil_model.conduct.shape[0]
        self.dz = dz
        self.ha,self.hs = -10000, 0
        self.A,self.B,self.C,self.F = np.zeros(self.N-1),np.zeros(self.N),np.zeros(self.N-1),np.zeros(self.N)
    
    def get_TriDiagonal(self,dt,h1,h2,tp,pond,flux):
        dz2 = np.power(self.dz, 2) 
        self.soil_model.calculate_props(h1,h2)
        s1,s2,k,cap = self.soil_model.theta1,self.soil_model.theta2,self.soil_molde.conduct,self.soil_model.capacity
        n = s1.shape[0]
        self.root_model.calculate_sink_source(h2,tp)
        sink = self.root_model.sink
        self.B[0] =1; self.C[0] = 0; self.F[0] = 0;# bottom boundary condition. later ! have to implemented no flux boundary. 
        if (h2[n-1]<self.ha) and (flux>=0): # lower atmospheric boundary condition if lower than this switch to dirichlet boundary
            self.B[n-1] =1; self.A[n-1] = 0; self.F[n-1] = 0
            h2[n-1] = self.ha
            # if flux < 0 it turns back to neumann type because of rainfall!
        if (pond > 0) and (flux< 0): #ponding boundary again switch to dirichlet boundary if pond > 0 solver has to check 0 < pond <= pond_max
            self.B[n-1] =1; self.A[n-1] = 0; self.F[n-1] = 0
            h2[n-1] = pond # stil raining and it can not hold more water. pond always <= pond_max
        
        else:
            k11 = (k[n-1] + k[n-2]) / 2 # averaging it could geometric!
            self.A[n-2] = -k11 / dz2
            self.B[n-1] =  cap[self.N-1] / dt + k11 / np.pow(self.dz, 2)
            self.F[n-1] = h2[n-2] * k11 / dz2 - h2[n-1] *  k11 / dz2 - k11/self.dz+ (s1[n-1]-s2[n-1]) / dt -flux/self.dz - sink[n-1]

        for i in range(1,n-1):
            k1 = (k[i] + k[i - 1]) / 2
            k2 = (k[i] + k[i + 1]) / 2
            self.A[i-1] = -k1 / dz2
            self.B[i] = ((k1 + k2) / dz2 + cap[i] / dt)
            self.C[i] = -k2 / dz2
            self.F[i] = (s1[i] - s2[i]) / dt + k2 * (h2[i + 1] - h2[i]) / dz2 - k1 * (h2[i] - h2[i - 1]) / dz2+ (k2 - k1) / self.dz - sink[i]
        
        return solve_thomas(self.A,self.B,self.C,self.F) + h2 #returning to new solution!