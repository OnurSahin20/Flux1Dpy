from numba.types import unicode_type, DictType
from numba import float64,int32
from numba.experimental import jitclass
import numpy as np 

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
        elif (self.method == 1) or (self.method==4): # van genuchten
            self.a,self.n,self.m,self.L,self.tr,self.ths,self.ks = self.pars["a"],self.pars["n"],self.pars["m"],self.pars["L"],self.pars["tr"],self.pars["ths"],self.pars["ks"]
        elif (self.method == 2) or (self.method==3): # FXW or FXW-M1 
            self.a,self.n,self.m,self.L,self.ths,self.ks = self.pars["a"],self.pars["n"],self.pars["m"],self.pars["L"],self.pars["ths"],self.pars["ks"]
        else:
            raise ValueError('Soil Hydraulic model options are BC, VGM, FXW or FXW-M1')
        
        N = self.ks.shape[0]
        self.theta1,self.theta2,self.conduct,self.capacity = np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)

    def correct_func(self, h, i):
        return np.power(np.log(np.e + np.power(np.abs(self.a[i] * h), self.n[i])), -self.m[i])
    
    def soil_moisture(self, h, i):
        if self.method == 0: #BC
            if h >= self.hb[i]:
                return self.ths[i]
            else:
                return self.tr[i] + (self.ths[i] - self.tr[i]) * (self.hb[i] / h)**self.lamb[i]
        
        elif self.method == 1: #VGM
            if h >= 0: 
                return self.ths[i]
            else:
                ah = np.abs(self.a[i] * h)
                return (self.ths[i] - self.tr[i]) / np.power(1 + np.power(ah, self.n[i]), self.m[i]) + self.tr[i]
        
        elif self.method == 2: #FXW
            return (1 - np.log(1 + h / self.hr) / np.log(1 + self.h0 / self.hr)) * self.correct_func(h, i) * self.ths[i]
        
        elif self.method == 3: #FXW-M1
            if h < self.hs:
                return self.ths[i] * (1 - np.log(1 + (h - self.hs) / self.hr) / np.log(1 + (self.h0 - self.hs) / self.hr)) * self.correct_func(h, i) / self.correct_func(self.hs, i)
            else:
                return self.ths[i]
                
        elif self.method == 4:  # VGM-AE (Vogel's Modified van Genuchten)
            hs = -2.0  # Air-entry value
            if h >= hs:
                return self.ths[i]
            else:
                ah_s = np.abs(self.a[i] * hs)
                se_s = np.power(1.0 + np.power(ah_s, self.n[i]), -self.m[i])
                thm = self.tr[i] + (self.ths[i] - self.tr[i]) / se_s
                ah = np.abs(self.a[i] * h)
                se_star = np.power(1.0 + np.power(ah, self.n[i]), -self.m[i])
                return self.tr[i] + (thm - self.tr[i]) * se_star

    def hydraulic_conductivity(self, h, i):
        if self.method == 0:
            if h >= self.hb[i]:
                return self.ks[i]
            else:
                return self.ks[i] * (self.hb[i] / h)**(2.0 + 3.0 * self.lamb[i])
    
        elif self.method == 1:
            if h >= 0: 
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
            if h < self.hs:
                rh = self.correct_func(h, i)
                r0 = self.correct_func(self.h0, i)
                rs = self.correct_func(self.hs, i)
                nom = 1 - np.power(1 - np.power(rh, 1 / self.m[i]), 1 - 1 / self.n[i])
                denom = 1 - np.power(1 - np.power(rs, 1 / self.m[i]), 1 - 1 / self.n[i])
                return self.ks[i] * np.power((rh - r0) / (rs - r0), self.L[i]) * np.power(nom / denom, 2)
            else: 
                return self.ks[i]
                
        elif self.method == 4:  # VGM-AE (Vogel's Modified van Genuchten)
            hs = -2.0
            if h >= hs:
                return self.ks[i]
            else:
                ah_s = np.abs(self.a[i] * hs)
                se_s = np.power(1.0 + np.power(ah_s, self.n[i]), -self.m[i])
                ah = np.abs(self.a[i] * h)
                se_star = np.power(1.0 + np.power(ah, self.n[i]), -self.m[i])
                term_num = (se_star ** self.L[i]) * (1.0 - (1.0 - se_star ** (1.0 / self.m[i])) ** self.m[i]) ** 2
                term_den = (se_s ** self.L[i]) * (1.0 - (1.0 - se_s ** (1.0 / self.m[i])) ** self.m[i]) ** 2
                return self.ks[i] * (term_num / term_den)

    def calculate_capacity(self, h, i): 
        if self.method == 0:
            if h >= self.hb[i]:
                return 0
            return -(self.lamb[i] * (self.ths[i] - self.tr[i]) / h) * (self.hb[i] / h)**self.lamb[i]

        elif self.method == 1:
            if h >= 0.0:
                return 0
            ah = self.a[i] * np.abs(h)
            numerator = self.a[i] * self.m[i] * self.n[i] * (self.ths[i] - self.tr[i]) * (ah**(self.n[i] - 1.0))
            denominator = (1.0 + ah**self.n[i])**(self.m[i] + 1.0)
            return numerator / denominator
            
        elif self.method in [2, 3]:
            if h >= 0:
                return 0
            else:
                f0 = self.soil_moisture(h, i)
                dh = h * np.power(10, -3)
                f1 = self.soil_moisture(h + dh, i)
                return (f1 - f0) / dh
                
        elif self.method == 4:  # VGM-AE
            hs = -2.0
            if h >= hs:
                return 10**-3# Prevents division-by-zero singularities in the matrix
            else:
                ah_s = np.abs(self.a[i] * hs)
                se_s = np.power(1.0 + np.power(ah_s, self.n[i]), -self.m[i])
                thm = self.tr[i] + (self.ths[i] - self.tr[i]) / se_s
                ah = self.a[i] * np.abs(h)
                numerator = self.a[i] * self.m[i] * self.n[i] * (thm - self.tr[i]) * (ah**(self.n[i] - 1.0))
                denominator = (1.0 + ah**self.n[i])**(self.m[i] + 1.0)
                return numerator / denominator            
    
    def calculate_props(self,h1,h2):
        n = h1.shape[0]
        for i in range(0,n):
            self.theta1[i] = self.soil_moisture(h1[i],i)
            self.theta2[i] = self.soil_moisture(h2[i],i)
            self.conduct[i] = self.hydraulic_conductivity(h2[i],i)
            self.capacity[i] = self.calculate_capacity(h2[i],i)
    
    def get_errors(self, h_old, h_new):
        theta_err_max = 0.0
        head_err_max = 0.0
        for i in range(h_old.shape[0]):
            theta_err = abs( self.soil_moisture(h_old[i], i)- self.soil_moisture(h_new[i], i))
            if theta_err > theta_err_max:
                theta_err_max = theta_err
            head_err = abs(h_old[i] - h_new[i])
            if head_err > head_err_max:
                head_err_max = head_err     
        return theta_err_max, head_err_max
    
    def only_moisture(self,h):
        n= h.shape[0]
        out = np.zeros(n)
        for i in range(0,n):
            out[i] = self.soil_moisture(h[i],i)
        return out
    
    def calculate_darcy(self,dz,h):
        n = h.shape[0]
        k = (self.hydraulic_conductivity(h[n-1],n-1) + self.hydraulic_conductivity(h[n-2],n-2)) / 2
        return - k * ((h[n-1]-h[n-2])/dz + 1)
    
    def calculate_pond(self,h,pond_old,dt,dz_top,flux_top,pond_max):
        qdarcy = self.calculate_darcy(dz_top,h)
        
        pond_new = pond_old + (-flux_top + qdarcy) * dt

        if pond_new >= pond_max:
            return pond_max
        if pond_new < 0:
            return 0
        else:
            return pond_new