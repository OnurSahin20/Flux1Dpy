
from numba.experimental import jitclass
from numba import float64,int32
import numpy as np 
from soil_model import SoilModels
from tridiagonal_varying_dz import CreateTriDiagonal
from source_sink import RootWaterUptake

soil_model_type = SoilModels.class_type.instance_type
tridiagonal_type = CreateTriDiagonal.class_type.instance_type
root_type= RootWaterUptake.class_type.instance_type
spec = {"soil_model":soil_model_type,"diagonal_model":tridiagonal_type, "root_model":root_type,
        "sim_time":float64,"sim_temp":float64,"dt_min":float64,"ini_head":float64[:],
        "transp":float64[:], 'dt_max':float64, 'ha':float64, 'hs':float64,
        'hnew':float64[:],'dt':float64,"dt_new":float64,'stat':int32,
        "flux_top":float64[:],'flux_bot':float64[:], 'top_bound':int32,
        "head_top":float64[:],'head_bot':float64[:]}

@jitclass(spec)
class NumericSolver:
    def __init__(self,soil_model,diagonal_model,root_model,sim_time,sim_temp,ini_head,flux_top,flux_bot,head_top,head_bot,transp):
        self.soil_model,self.diagonal_model,self.root_model = soil_model,diagonal_model,root_model #other class type inputs
        self.sim_time,self.sim_temp = sim_time,sim_temp # controlling simulation time
        self.ini_head = ini_head # initialized head!
        self.hnew = np.zeros(self.ini_head.shape)
        self.hnew[:] = self.ini_head[:]
        self.transp = transp
        self.dt_min, self.dt_max,self.dt,self.dt_new = 1 / 60,144.0,1.5,1.5
        self.stat = 1
        self.flux_top,self.flux_bot = flux_top,flux_bot
        self.head_top,self.head_bot = head_top,head_bot
        self.top_bound = self.diagonal_model.top_bound
        self.ha = self.diagonal_model.ha
        self.hs = self.diagonal_model.pond_max

    def control_dt(self,dt,i):
        if (i<=3):
            if dt * 1.2 < self.dt_max:
                return dt * 1.2
            else:
                return self.dt_max
        elif (i > 7):
            if (dt * 0.8 > self.dt_min):
                return dt * 0.7
            else:
                return self.dt_min
        else:
            return dt 
        
    def IterateTime(self,index,atmosp):
        eps_sm = 10 ** -6
        eps_h = 0.1
        self.hnew[:] = self.ini_head[:]
        self.stat = 0.5
        for i in range(0,15,1):
            hx = self.diagonal_model.get_new(self.dt, self.ini_head,self.hnew,self.flux_top[index],self.flux_bot[index],
                                             self.head_top[index],self.head_bot[index],self.transp[index],atmosp)
        
            err_sm,err_h = self.soil_model.get_errors(self.hnew,hx)
            self.hnew[:] = hx[:] 
            if (err_sm <= eps_sm ) and (err_h <=  eps_h):
                 self.stat = 0
                 break
        self.dt_new = self.control_dt(self.dt,i)
        return hx 
        
    def RunSolver(self):
        r,c = self.flux_top.shape[0],self.ini_head.shape[0]
        hout,sout = np.zeros((r+1,c)), np.zeros((r+1,c))
        
        hout[0,:] = self.ini_head[:]
        n1 = hout.shape[0]-1
        sout[0,:] = self.soil_model.only_moisture(self.ini_head)[:]
        count_time, ind_time= 0.0,0.0
        index = int(0)
        while (count_time<self.sim_time):
            save_time = self.sim_temp - ind_time
            if self.dt > save_time:
                self.dt = save_time
           
            hnew = self.IterateTime(index,1)
            if (self.top_bound==4):
                if (self.ha <= hnew[n1]<=self.hs):
                    x = 5 # dummy 
                else:
                    hnew = self.IterateTime(index,0)
            
            if self.stat != 0:
                self.dt = self.dt / 2
                if self.dt < self.dt_min:
                    raise ValueError(f"Solver failed to converge. dt dropped below {self.dt_min}")

            else:
                self.dt = self.dt_new 
                self.ini_head[:] = hnew[:]
                count_time += self.dt
                ind_time += self.dt

            if ind_time >= self.sim_temp:
                hout[index+1,:] = self.ini_head[:]
                sout[index+1,:] = self.soil_model.only_moisture(self.ini_head)
                ind_time = ind_time - self.sim_temp
                index +=1 
        return hout