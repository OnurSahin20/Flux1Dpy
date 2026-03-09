from numba.types import unicode_type, DictType
from numba import float64,int32
from numba.experimental import jitclass
import numpy as np 


# 1. Define the spec with unicode_type for the string
spec = {
    'method': int32,'pars': DictType(unicode_type, float64[:]),
    "p0":float64, 'p0opt':float64, 'p2h':float64, 'p2l':float64, 'p3':float64, 'r2h':float64, 'r2l':float64, #feddes parameters
    'p50':float64,'p0':float64, # s- shape parameter alpha calculation
    'tranp':float64, #transpiration for that time.
    'bx':float64[:], 'alfa':float64[:]
    }


@jitclass(spec)
class SoilModels:
    def __init__(self,method,pars):
        self.method = method
        self.pars = pars
        if self.method == 0:
            self.p50,self.p0 = self.pars['p50'],self.pars['p0']
        elif self.method ==1:
            self.p0, self.p0opt, self.p2h, self.p2l, self.p3= self.pars['p0'],self.pars['p0opt'],self.pars['p2h'],self.pars['p2l'],self.pars['p3']
            self.r2h, self.r2l  = self.pars['r2h'],self.pars['r2l']
        
        self.bx = self.pars['bx']
        self.tp = self.pars['transp']
        self.sink  = np.zeros(self.bx.shape)
    
    def alpha_feddes(self,h):
      
        if self.tp > self.r2h:
            p2 = self.p2h
        elif self.tp < self.r2l:
            p2 = self.p2l
        else:
            p2 = self.p2l + (self.p2h - self.p2l) * (self.tp - self.r2l) / (self.r2h - self.r2l)

        if h >= self.p0 or h <= self.p3:
            return 0
        elif self.p0opt <= h < self.p0:
            return (h - self.p0) / (self.p0opt - self.p0)
        elif self.p0opt > h and h >= p2:
            return 1
        elif self.p3 <= h < p2:
            return (h - self.p3) / (self.p2 - self.p3)
        else:
            return 0
    
    def alfa_sshape(self,h): 
        return 1 / (1 + np.power(h/self.p50,self.p0))
        
    def calculate_sink_source(self,h,tp):
        for i in range(h.shape[0]):
            if self.method == 0:
                self.sink[i] = self.alfa_sshape(h[i]) * self.bx[i] * tp
            else:
                self.sink[i] = self.alpha_feddes(h[i],tp) * self.bx[i] * tp 