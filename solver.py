
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
        'hnew':float64[:],'dt':float64,"dt_new":float64,'stat':int32, 'n1':int32,
        "flux_top":float64[:],'flux_bot':float64[:], 'top_bound':int32,
        "head_top":float64[:],'head_bot':float64[:]}

@jitclass(spec)
class NumericSolver:
    def __init__(self,soil_model,diagonal_model,root_model,sim_time,sim_temp,ini_head,flux_top,flux_bot,head_top,head_bot,transp):
        self.soil_model,self.diagonal_model,self.root_model = soil_model,diagonal_model,root_model
        self.sim_time,self.sim_temp = sim_time,sim_temp
        self.hnew = np.zeros(ini_head.shape)
        self.ini_head = np.zeros(ini_head.shape)
        self.ini_head[:] = ini_head[:]
        self.transp = transp
        self.dt_min, self.dt_max,self.dt,self.dt_new = 1 / 60,self.sim_temp,1.5,1.5
        self.stat = 1
        self.flux_top,self.flux_bot = flux_top,flux_bot
        self.head_top,self.head_bot = head_top,head_bot
        self.top_bound = self.diagonal_model.top_bound
        self.ha = self.diagonal_model.ha
        self.hs = self.diagonal_model.pond_max
        self.n1 = self.ini_head.shape[0] - 1

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
    
    def IterateTime(self, index, pond): # 1. Pass pond as an argument
        eps_sm,eps_h = 10 ** -3,1
        self.hnew[:] = self.ini_head[:]
        self.stat = 1
        if pond > 0:
            current_atmosp = 0
            dirichlet = 1
            self.hnew[self.n1] = pond
        else:
            current_atmosp = 1
            dirichlet = 0
            
        i = 0
        while (i < 10):
            hx = self.diagonal_model.get_new(self.dt, self.ini_head, self.hnew,self.flux_top[index], self.flux_bot[index],
                                             self.head_top[index], self.head_bot[index],self.transp[index], current_atmosp)
        
            if (self.top_bound == 4):
                if (dirichlet == 0):
                    if (self.ha < hx[self.n1] < 0):          
                        current_atmosp = 1
                    else:
                        current_atmosp = 0
                        dirichlet = 1
                        self.hnew[:] = self.ini_head[:]
                        if hx[self.n1] >= 0:
                            self.hnew[self.n1] = 0
                        else:
                            self.hnew[self.n1] = self.ha
                        i = 0 
                        continue
                else:
                    if hx[self.n1] >= 0:
                        self.hnew[self.n1] = pond 
                    else:
                        self.hnew[self.n1] = self.ha
            i += 1            
            err_sm, err_h = self.soil_model.get_errors(self.hnew, hx)
            self.hnew[:] = hx[:] 
        
            if (err_sm <= eps_sm) and (err_h <= eps_h):
                 self.stat = 0
                 break
            
        self.dt_new = self.control_dt(self.dt, i)
        return hx
    
    def RunSolver(self, time_interval=1):
        count_time, ind_time, index, pond = 0.0, 0.0, int(0), 0
        dz_top = self.diagonal_model.dz_top
        r, c = self.flux_top.shape[0], self.ini_head.shape[0]
        
        # 1. Shrink the arrays to save memory immediately
        out_rows = (r // time_interval) + 1
        hout, sout, sink_out = np.zeros((out_rows, c)), np.zeros((out_rows, c)), np.zeros((out_rows, c))
        
        out_idx = 0  # Separate tracker for our smaller output array
        
        # Save initial state
        hout[out_idx, :] = self.ini_head[:]
        sout[out_idx, :] = self.soil_model.only_moisture(self.ini_head)[:]
        self.root_model.calculate_sink_source(self.ini_head, self.transp[index])  
        sink_out[out_idx, :] = self.root_model.sink
        
        out_idx += 1

        while (count_time < self.sim_time):
            save_time = self.sim_temp - ind_time
            if self.dt > save_time:
                self.dt = save_time
           
            hnew = self.IterateTime(index, pond)
            if self.stat != 0:
                self.dt = self.dt / 3
                if self.dt < self.dt_min:
                    raise ValueError(f"Solver failed to converge. dt dropped below {self.dt_min}")

            else:
                if hnew[self.n1] >= 0: 
                    pond = self.soil_model.calculate_pond(hnew, pond, self.dt, dz_top, self.flux_top[index], self.hs)
                else:
                    pond = 0
                
                count_time += self.dt
                ind_time += self.dt
                self.dt = self.dt_new 
                self.ini_head[:] = hnew[:]  
                print(count_time,self.ini_head[-1])
                if ind_time >= self.sim_temp:
                    # 2. Only save to the array if we hit the requested interval
                    if (index + 1) % time_interval == 0:
                        hout[out_idx, :] = self.ini_head[:]
                        sout[out_idx, :] = self.soil_model.only_moisture(self.ini_head)
                        self.root_model.calculate_sink_source(self.ini_head, self.transp[index])  
                        sink_out[out_idx, :] = self.root_model.sink
                        out_idx += 1

                    ind_time = ind_time - self.sim_temp
                    index += 1
                    
        return hout, sout, sink_out