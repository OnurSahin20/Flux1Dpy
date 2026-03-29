import numpy as np
from numba.typed import Dict
from numba import types
from soil_model import SoilModels
from source_sink import RootWaterUptake
from tridiagonal_varying_dz import CreateTriDiagonal

class InfiltrationModel:
    def __init__(self, sim_time: int, temp_time:int,discrete: dict[str:np.ndarray, str:float]) -> None:
        self.sim_time, self.temp_time = sim_time,temp_time
        self.discrete = discrete
        self.z,self.nodes = self.setup_grid()
        self.numba_soil,self.numba_root, self.numba_tridiagonal, self.numba_solver  = None, None, None, None

    def setup_grid(self):
        z_list,z = [0.0], 0.0
        for layer_len, dz in zip(self.discrete["layers"], self.discrete["dz"]):
            num_cells = int(round(layer_len / dz)) 
            for _ in range(num_cells):
                z += dz
                z_list.append(z)    
        return np.array(z_list),len(z_list) 
    
    def set_soil_model(self,hydraulic_model,soil_data) -> None:
        options = {'BC':0,"VGM":1,'FXW':2,'FXW-M1':3} # converting string hydraulic model to integer for numba speed!
        if hydraulic_model== "VGM":
            req_params = ["a","n","m","tr","ths","ks","L"]
        elif hydraulic_model == "BC":
            req_params = ['lamb','hb',"tr","ths","ks"]
    
        elif hydraulic_model == "FXW" or hydraulic_model == "FXW-M1":
            req_params = ["a","n","m","ths","ks","L"] 
        else:
            raise ValueError("Models BC,VGM, FXW, FXW-M1 are supported!")
        
        if set(soil_data.keys()) != set(req_params):
            raise ValueError(f'Model {hydraulic_model} parameters should be {req_params} but given as {soil_data.keys()}')
        
        self.hydro_model = hydraulic_model
        key_type,value_type = types.unicode_type,types.float64[::1] 
        self.soil_params = Dict.empty(key_type, value_type)
        for param in soil_data.keys():
            profile = self.create_vertical_profile(np.array(soil_data[param]))
            self.soil_params[param] = np.ascontiguousarray(profile, dtype=np.float64)
        
        self.numba_soil= SoilModels(options[self.hydro_model],self.soil_params)

    def set_root_model(self,root_model = "",root_params={},
                       root_distribution="", root_depth = 0):
        options = {'s-shape':0,'feddes':1} # converting string hydraulic model to integer for numba speed!
        self.root_model = root_model
        self.root_params,self.root_dist = root_params,root_distribution
        self.rzl = root_depth
        key_type,value_type = types.unicode_type,types.float64 
        self.root_params = Dict.empty(key_type, value_type)
        for param in root_params.keys():
            self.root_params[param] = root_params[param] 
        bx = self.create_root_distribution()
        self.numba_root = RootWaterUptake(options[self.root_model],self.root_params,bx)
    
    def set_boundary_conditions(self,hini,top_bound,bot_bound,flux_top=0,flux_bot=0,ponding_max=0) -> None: # handling here boundary condition
        self.top_opts = {'constant head':0,'constant_flux':1,'variable head':2,'variable flux':3,'atmospheric':4}
        self.bot_opts = {'constant head':0,'constant_flux':1,'variable head':2,'variable flux':3,'free drainage':4,'seepage face':5}
        self.hini = hini
        self.flux_top,self.flux_bot = flux_top,flux_bot # either time series if it is time dependent or one variable for fixed
        self.ponding  = ponding_max # if it is bigger than 0, ponding is occurs in the top boundary!
        self.check  = int(self.sim_time / self.temp_time)
        if self.top_bound in [2,3,4]:
            if (self.check != self.flux.shape[0]) or (self.check != self.transpiration.shape[0]):
                raise ValueError(f'Simulation time and size of vectors should have to consistent for boundary {self.top_bound}')
  
        self.numba_tridiagonal  = CreateTriDiagonal(self.numba_soil_class,self.numba_root_class,self.dz,self.top_opts[top_bound],self.bot_opts[bot_bound])

    def create_vertical_profile(self, x: list) -> np.ndarray:
        vertic_prof = np.zeros(self.nodes)
        c,count = 0,0.0    
        for i in range(self.nodes):
            vertic_prof[i] = x[c]
            if i < self.nodes - 1:
                current_dz = self.discrete["dz"][c]
                count += current_dz
                if c < len(x) - 1 and count >= self.discrete["layers"][c] - 1e-9:
                    c += 1
                    count = 0.0           
        return vertic_prof

    def create_root_distribution(self) -> np.ndarray:
        bx = np.zeros(self.nodes, dtype=np.float64)
        if self.root_dist == "normalized":
            z = 0
            for i in range(self.nodes):
                if z < self.z - self.rzl:
                    bx[i] = 0
                elif z > self.z - 0.2 * self.rzl:
                    bx[i] = 1.667 / (self.rzl / self.dz)
                else:
                    bx[i] = 2.0833 / (self.rzl / self.dz) * (1 - (self.z - z) / self.rzl)
                z += self.dz
        elif self.root_dist == "equally":
            b = 1 / int(self.rzl / self.dz)
            bx[self.nodes - int(self.rzl / self.dz):] = b
        return bx



        

