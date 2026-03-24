
from numba.experimental import jitclass
from numba import float64,int32
import numpy as np 
from soil_model import SoilModels
from tridiagonal import CreateTriDiagonal
from source_sink import RootWaterUptake



soil_model_type = SoilModels.class_type.instance_type
tridiagonal_type = CreateTriDiagonal.class_type.instance_type
root_type= RootWaterUptake.class_type.instance_type
spec = {"soil_model":soil_model_type,"diagonal_model":tridiagonal_type, "root_model":root_type,
        "sim_time":float64,"sim_temp":float64,"dt_min":float64,"ini_head":float64[:],
        "flux":float64[:],"transp":float64[:],"pond_max":float64, 'dt_max':float64,
        'hnew':float64[:],'dt':float64,"dt_new":float64,'stat':int32,'dz':float64}


@jitclass(spec)
class NumericSolver:
    def __init__(self,soil_model,diagonal_model,root_model,sim_time,sim_temp,dz,ini_head,flux,transp,pond_max):
        self.soil_model,self.diagonal_model,self.root_model = soil_model,diagonal_model,root_model #other class type inputs
        self.sim_time,self.sim_temp = sim_time,sim_temp # controlling simulation time
        self.ini_head = ini_head # initialized head!
        self.hnew = np.zeros(self.ini_head.shape)
        self.hnew[:] = self.ini_head[:]
        self.flux,self.transp = flux,transp # meteorological data!
        self.pond_max = pond_max # if > 0 ponding occurs!
        self.dt_min, self.dt_max,self.dt,self.dt_new = 1 / 60,self.sim_temp,1.5,1.5
        self.stat = 1 # controls solver is successful or not flag
        self.dz = dz
        
    

    def control_dt(self,dt,i):
        if (i<=3):
            if dt * 1.3 < self.dt_max:
                return dt * 1.3
            else:
                return self.dt_max
        elif (i > 7):
            if (dt * 0.8 > self.dt_min):
                return dt * 0.8
            else:
                return self.dt_min
        else:
            return dt 
        
    def IterateTime(self,hold,flux,tp,dt,pond):
        eps,mass_error = 10 ** -4,10 # accuracy and initializing the mass error!
        self.hnew[:] = hold[:]
        n = self.hnew.shape[0] - 1
        self.stat = 1 
        for i in range(0,15,1):
            self.hnew[0] = 0 #water table
            hx = self.diagonal_model.get_new(dt,hold,self.hnew,tp,pond,flux)
            mass_error = self.soil_model.get_error(self.hnew,hx)
            self.hnew[:] = hx[:] 
            
            if (mass_error<=eps):
                 self.stat = 0
                 break
        
        self.dt_new = self.control_dt(dt,i)
        return hx 
        
        

    def RunSolver(self):
        r,c = self.flux.shape[0],self.ini_head.shape[0]
        hout,sout = np.zeros((r+1,c)), np.zeros((r+1,c))
        hout[0,:] = self.ini_head[:]
        sout[0,:] = self.soil_model.only_moisture(self.ini_head)[:]
        pond = 0.0
        count_time, ind_time= 0.0,0.0
        index = int(0)
        while (count_time<self.sim_time):
            hnew = self.IterateTime(self.ini_head,self.flux[index],self.transp[index],self.dt,pond)
            
            if self.stat != 0:
                self.dt = self.dt / 2 
                 
            else:
                
                count_time += self.dt
                ind_time += self.dt
                if self.pond_max > 0:
                    pond = self.soil_model.calculate_pond(self.dz,hnew,self.dt,pond,self.flux[index])
            
                if pond > self.pond_max:
                    pond = self.pond_max
                
                self.dt = self.dt_new 
                self.ini_head[:] = hnew[:]

            if ind_time >= self.sim_temp:
                hout[index+1,:] = self.ini_head[:]
                sout[index+1,:] = self.soil_model.only_moisture(self.ini_head)
                ind_time = ind_time - self.sim_temp
                index +=1 