
from numba.experimental import jitclass
from numba import float64,int32
import numpy as np 
from soil_model import SoilModels
from tridiagonal import CreateTriDiagonal
from source_sink import RootWaterUptake

soil_model_type = SoilModels.class_type.instance_type
tridiagonal_type = CreateTriDiagonal.class_type.instance.type
root_type= RootWaterUptake.class_type.instance.type
spec = {"soil_model":soil_model_type,"diagonal":tridiagonal_type, "root_model":root_type,
        "sim_time":float64,"sim_temp":float64,"dt_min":float64,"ini_head":float64[:],
        "flux":float64[:],"transp":float64[:],"pond_max":float64}


class NumericSolver:
    def __init__(self,soil_model,diagonal,root_model,sim_time,sim_temp,ini_head,flux,transp,pond_max):
        self.model,self.diagonal,self.root_model = soil_model,diagonal,root_model #other class type inputs
        self.sim_time,self.sim_temp = sim_time,sim_temp # controlling simulation time
        self.ini_head = ini_head # initialized head!
        self.flux,self.transp = flux,transp # meteorological data!
        self.pond_max = pond_max # if > 0 ponding occurs!
    
    
