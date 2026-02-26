import numpy as np 

class SoilModels:
    def __init__(self,method:str,parameters:dict) ->None:
        #method is the hydraulic model VGM,FXW Brooks and Corey etc
        # parameters : dictonary of required parameter of given model for VGM = {a:np.array([a1,a2,...an]),ks:np.array([k1,k2,...kn])}
        self.method = method
        self.par = parameters
        self.hs,self.hr,self.h0 = -1, -1500,-0.63 * np.power(10,7)
        if self.method == "VGM":
            req_params = ["a","n","tr","ths","ks"]
            if set(self.par.keys()) != set(req_params):
                raise ValueError(f"Model {self.method} parameters should be {req_params} but given as {self.par.keys()}")
            sorted_dict = {k: self.par[k] for k in req_params}
            self.a,self.n,self.tr,self.ths,self.ks = sorted_dict.values()
            self.m = 1 - 1 / self.n
            self.L = np.array([0.5] * self.m.shape[0])
            if not (self.a.size == self.n.size == self.ths.size,self.tr.size==self.ks.size):
                raise ValueError(f"sizes do not match of the parameters!")

            

        elif self.method == "FXW" or self.method == "FXW-M1":
            req_params = ["a","n","m","ths","ks"]
            if set(self.par.keys()) != set(req_params):
                raise ValueError(f"Model {self.method} parameters should be {req_params} but given as {self.par.keys()}")
            if not (self.a.size == self.n.size == self.ths.size,self.m.size==self.ks.size):
                raise ValueError(f"sizes do not match of the parameters!")
            sorted_dict = {k: self.par[k] for k in req_params}
            self.a,self.n,self.m,self.ths,self.ks = self.par.values()
        else:
            raise ValueError("Models VGM, FXW, FXW-M1 are supported!")

        self.N = self.ks.shape[0]

    def correction_func(self,h) -> np.ndarray:
        # function for FXW and FXW-M1.
        if isinstance(h, float):
            return np.array([np.power(np.log(np.e + np.power(np.abs(self.a * h), self.n)), -self.m)] * self.N)
        else:
            return np.power(np.log(np.e + np.power(np.abs(self.a * h), self.n)), -self.m)
    
    def calculate_soil_moisture(self,h:np.ndarray) -> np.ndarray:
        ah = np.abs(self.a * h)
        if (self.method == "VGM"): 
            moist = (self.ths - self.tr) / np.power(1 + np.power(ah, self.n), self.m) + self.tr
            moist[h>=0] = self.ths[h>=0]
        elif (self.method == "FXW"):
             moist = (1 - np.log(1 + h / self.hr) / np.log(1 + self.h0 / self.hr)) * self.correct_func(h) * self.ths
        elif (self.method == "FXW-M1"):
             moist = self.ths * (1 - np.log(1 + (h - self.hs) / self.hr) / np.log(1 + (self.h0 - self.hs) / self.hr)) * self.correct_func(h) / np.array([self.correct_func(self.hs)] * self.N)
             moist[h<self.hs] = self.ths[h<self.hs]
            
        return moist
    
    def calculate_conductivity(self,h:np.ndarray) -> np.ndarray:
        if (self.method == "VGM"):
            se = np.power((1 + np.power(np.abs(self.a * h), self.n)), -self.m)
            k = self.ks * np.power(se, self.L) * np.power((1 - np.power((1 - np.power(se, (1 / self.m))), self.m)), 2)
        
        elif (self.method == "FXW"):
            rh = self.correct_func(h)
            r0 = self.correct_func(self.h0)
            sek = (rh-r0) / (1-r0)
            k = self.ks * np.power(sek,self.L) * np.power(1 - np.power(1 - np.power(rh,1/self.m),1-1/self.n),2); 
        elif (self.method =="FXW-M1"):
            rh,r0,rs = self.correct_func(h),self.correction_func(self.h0),self.correction_func(self.hs)
            nom = 1 - np.power(1 - np.power(rh, 1 / self.m), 1 - 1 / self.n)
            denom = 1 - np.power(1 - np.power(rs, 1 / np.m), 1 - 1 / self.n)
            k = self.ks * np.power((rh - r0) / (rs - r0), self.L) * np.power(nom / denom, 2)
            k[h<self.hs] = self.ks[h<self.ks]
        
        return k
    

if __name__ == "__main__":
    soil_properties = {"a": np.array([-0.33 / 100] * 100), "tr": np.array([0.08] * 100), "ths": np.array([0.45] * 100),
                   "ks": np.array([120 * 100 / 1440] * 100), "n": np.array([3.0]*100)}  # soil properties of Genuchten-VGM model
    
    soil_model = SoilModels("VGM",soil_properties)
    print(soil_model.calculate_conductivity(np.linspace(0,-1000,100)))
 