from numba.types import unicode_type, DictType
from numba import float64,int32
from numba.experimental import jitclass
import numpy as np 


# 1. Define the spec with unicode_type for the string
spec = {
    'method': int32,'pars': DictType(unicode_type, float64[:]),'hs': float64, 'h0': float64,'hr': float64,'lamb': float64[:],'hb': float64[:],
    'tr': float64[:],'ths': float64[:],'ks': float64[:],'a': float64[:],'n': float64[:], 'm': float64[:],  'L': float64[:],
    'theta1': float64[:],  'theta2': float64[:],  'conduct': float64[:],  'capacity': float64[:]}


@jitclass(spec)
class SoilModels:
    def __init__(self,method,pars):
        self.method = method
        self.pars = pars
        self.hs,self.h0,self.hr = -1,-0.63 * np.power(10,7),-1500
        if self.method == 0: #brooks and corey
            self.lamb,self.hb,self.tr,self.ths,self.ks =self.pars['lambda'],self.pars['hb'],self.pars['tr'],self.pars['ths'],self.pars['ks']
        elif self.method == 1: # van genuchten
            self.a,self.n,self.m,self.L,self.tr,self.ths,self.ks = self.pars["a"],self.pars["n"],self.pars["m"],self.pars["L"],self.pars["tr"],self.pars["ths"],self.pars["ks"]
        elif (self.method == 2) or (self.method==3): # FXW or FXW-M1 
            self.a,self.n,self.m,self.L,self.ths,self.ks = self.pars["a"],self.pars["n"],self.pars["m"],self.pars["L"],self.pars["ths"],self.pars["ks"]
        else:
            raise ValueError('Soil Hydraulic model options are BC, VGM, FXW or FXW-M1')
        
        N = self.ks.shape[0]
        self.theta1,self.theta2,self.conduct,self.capacity = np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)

    def correct_func(self, h, i):
        return np.power(np.log(np.e + np.power(np.abs(self.a[i] * h), self.n[i])), -self.m[i])
    
    def soil_moisture(self,h,i):
        if self.method == 0:
            if h >= self.hb[i]:
                return self.ths[i]
            else:
                return self.tr[i] + (self.ths[i] - self.tr[i]) * (self.hb[i] / h)**self.lamb[i]
        elif self.method ==1:
            if (h>=0): 
                return self.ths[i]
            else:
                ah = np.abs(self.a[i] * h)
                return (self.ths[i] - self.tr[i]) / np.power(1 + np.power(ah, self.n[i]), self.m[i]) + self.tr[i]
        elif self.method == 2:
            return (1 - np.log(1 + h / self.hr) / np.log(1 + self.h0 / self.hr)) * self.correct_func(h, i) * self.ths[i]
        
        elif self.method ==3:
            if (h < self.hs):
                return self.ths[i] * (1 - np.log(1 + (h - self.hs) / self.hr) / np.log(1 + (self.h0 - self.hs) / self.hr)) * self.correct_func(h, i) / self.correct_func(self.hs, i)
            else:
                return self.ths[i]
    
    def hydraulic_conductivity(self,h,i):

        if self.method == 0:
            if h >= self.hb[i]:
                return self.ks[i]
            else:
                return self.ks[i] * (self.hb[i] / h)**(2.0 + 3.0 * self.lamb[i])
    
        elif self.method ==1:
            if (h>=0): 
                return self.ks[i]
            else:
                se = np.power((1 + np.power(np.abs(self.a[i] * h), self.n[i])), -self.m[i])
                return self.ks[i] * np.power(se, self.L[i]) * np.power(
                        (1 - np.power((1 - np.power(se, (1 / self.m[i]))), self.m[i])), 2)
        
        elif self.method == 2:
            rh = self.correct_func(h,i)
            r0 = self.correct_func(self.h0,i)
            sek = (rh-r0) / (1-r0)
            return self.ks[i] * np.power(sek,self.L[i]) * np.power(1 - np.power(1 - np.power(rh,1/self.m[i]),1-1/self.n[i]),2)
        
        elif self.method == 3:
            if (h < self.hs):
                rh = self.correct_func(h, i)
                r0 = self.correct_func(self.h0, i)
                rs = self.correct_func(self.hs, i)
                nom = 1 - np.power(1 - np.power(rh, 1 / self.m[i]), 1 - 1 / self.n[i])
                denom = 1 - np.power(1 - np.power(rs, 1 / self.m[i]), 1 - 1 / self.n[i])
                return self.ks[i] * np.power((rh - r0) / (rs - r0), self.L[i]) * np.power(nom / denom, 2)
            else: 
                return self.ks[i]

    def calculate_capacity(self,h,i): 
        #The specific moisture capacity anaytical formulation for brooks and corey and genuchten and numerical derivative for FXW and FXW-M1
        if self.method == 0:
            if h >= self.hb[i]:
                return 0.0
            return -(self.lamb[i] * (self.ths[i] - self.tr[i]) / h) * (self.hb[i] / h)**self.lamb[i]

        elif self.method == 1:
            if h >= 0.0:
                return 0.0
            ah = self.a[i] * np.abs(h)
            numerator = self.a[i] * self.m[i] * self.n[i] * (self.ths[i] - self.tr[i]) * (ah**(self.n[i] - 1.0))
            denominator = 1.0 + ah**self.n[i]**(self.m[i] + 1.0)
            return numerator / denominator
        else:
            if h >=0:
                return 0
            else:
                f0 = self.soil_moisture(h, i)
                dh = h * np.power(10, -3)
                f1 = self.soil_moisture(h + dh, i)
                return (f1-f0) / dh
            
    def calculate_props(self,h1,h2):
        n = h1.shape[0]
        for i in range(0,n):
            self.theta1[i] = self.soil_moisture(h1[i],i)
            self.theta2[i] = self.soil_moisture(h2[i],i)
            self.conduct[i] = self.hydraulic_conductivity(h2[i],i)
            self.capacity[i] = self.calculate_capacity(h2[i],i)