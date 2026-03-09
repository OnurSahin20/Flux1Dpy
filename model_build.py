import numpy as np
from numba.typed import Dict
from numba import types

class InfiltrationModel:
    def __init__(self, sim_time: int, temp_time:int,discrete: dict[str:np.ndarray, str:float],
                 hydraulic_model: str,soil_data: dict[str, list], flux: np.ndarray,transpiration:np.ndarray,
                 plant_function="",root_params={},root_distribution="",
                 root_depth = 0,ponding_max=0) -> None:
        
        self.sim_time, self.temp_time = sim_time,temp_time
        self.hydro_model = hydraulic_model
        self.discrete = discrete
        self.flux = flux
        self.z = sum(self.discrete["layers"])
        self.lay,self.dz = self.discretize_soil_colum_1d()
        if hydraulic_model== "VGM":
            req_params = ["a","n","m","tr","ths","ks","L"]
            if set(sorted(soil_data.keys())) != set(req_params):
                raise ValueError(f"Model {hydraulic_model} parameters should be {req_params} but given as {soil_data.keys()}")
        
        elif hydraulic_model == "FXW" or hydraulic_model == "FXW-M1":
            req_params = ["a","n","m","ths","ks","L"]
            if set(soil_data.keys()) != set(req_params):
                raise ValueError(f"Model {hydraulic_model} parameters should be {req_params} but given as {soil_data.keys()}")
            
        else:
            raise ValueError("Models VGM, FXW, FXW-M1 are supported!")

    
        self.soil_data_dict,self.time_series,self.ponding,self.plant_func  = soil_data,flux,ponding_max,plant_function
        self.root_params,self.root_dist,self.transpiration = root_params,root_distribution,transpiration
        self.rzl = root_depth

    def discretize_soil_colum_1d(self):
        Z,N = sum(self.discrete["layers"]),self.discrete["num_nodes"]
        
        if self.discrete["uniform"]:
            return np.linspace(0,Z,N+1),np.array([Z/N] * (N+1))
        else:
            raise ValueError("Manually passing the z array not implemented yet!")

    
    def create_vertical_profile(self, x: np.ndarray) -> np.ndarray:
        """function returns vertical soil properties. Direction is upward!!!!"""
        vertic_prof,c,count = np.zeros(self.lay.shape),0,0.0
        for i in range(vertic_prof.shape[0]):
            vertic_prof[i] = x[c]
            if i < len(self.dz):
                count += self.dz[i]
                if c < len(x) - 1 and count >= self.discrete["layers"][c]:
                    c += 1
                    count = 0.0
        return vertic_prof

    def get_soil_properties(self) -> Dict: #numba type dict 
        """Method gathers vertical features of layers into a Numba-typed dictionary."""
        key_type,value_type = types.unicode_type,types.float64[::1] 
        soil_params = Dict.empty(key_type, value_type)
        for param in self.soil_data_dict.keys():
            profile = self.create_vertical_profile(np.array(self.soil_data_dict[param]))
            soil_params[param] = np.ascontiguousarray(profile, dtype=np.float64)
        return soil_params

    def get_root_params(self) ->Dict:
        key_type,value_type = types.unicode_type,types.float64 
        params = Dict.empty(key_type, value_type)
        for param in self.root_params.keys():
            params[param] = self.root_params[param] 
        return params
          
    def create_root_distribution(self) -> np.ndarray:
        n = self.lay.shape[0]
        bx = np.zeros(n,dtype=np.float64)
        if self.root_dist == "normalized":
            z = 0
            for i in range(n):
                if z < self.z - self.rzl:
                    bx[i] = 0
                elif z > self.z - 0.2 * self.rzl:
                    bx[i] = 1.667 / (self.rzl / self.dz[i])
                else:
                    bx[i] = 2.0833 / (self.rzl / self.dz[i]) * (1 - (self.z - z) / self.rzl)
                z += self.dz[i]
        elif self.root_dist == "equally":
            b = 1 / int(self.rzl / self.dz[0])
            bx[n - int(self.rzl / self.dz[0]):] = b
        
        return bx



        

