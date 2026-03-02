import os
import numpy as np

"""
Definition of the arguments in Infiltration Model
length unit is always cm.
time unit can be "hour" or "day".
discrete:dict is the number of layers and length of the layers. Direction is positive upward. [0] index is bottom Example: 
discretize = {"layers": [50, 50], "dz": 1 cm} 50 cm each two layer and dz cell size 1 cm total n = 100 cell
hydraulic_model:str currently supports VGM (Van-Genuchten), FXW and FXW-M1.
soil_data = {"soil_prop:[p1, p2, p3, ... pl]"} hydraulic properties of the soil model. Example;
soil_properties = {"alpha (1/cm)": [0.075, 0.019],
                   "tr": [0.065, 0.095], "ths": [0.41, 0.41], "Ks (cm/min)": [106.1 / 1440, 6.24 / 1440],
                   "n": [1.89, 1.31], "m": [1-1/1.89,1-1/1.31], "L": [0.5, 0.5]}
flux is net infiltration or exfiltration rate from top of the soil. (precipitation - evaporation). It should be negative
if precipitation bigger than evaporation because direction is positive upward.
initial:list is the initial condition of hydraulic pressure len should be int(z/dz) example:
n = int(z/dz), initial = [-100] * n
pond_max:float is the maximum allowable ponding as reservoir on the top of the surface. If it is zero no ponding
and extra water becomes runoff
"""

class InfiltrationModel:
    def __init__(self, discrete: dict[str:np.ndarray, str:float],
                 hydraulic_model: str,
                 soil_data: dict[str, list], flux: np.ndarray,sim_time: int, ) -> None:
                
 
        self.sim_time = sim_time
        self.discrete = discrete
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

    
        self.soil_data_dict = soil_data
        self.time_series = flux
      
    def discretize_soil_colum_1d(self):
        Z,N,dz_min = sum(self.discrete["layers"]),self.discrete["num_nodes"],self.discrete["dz_min"]
        
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

    def get_soil_properties(self) -> dict:
        """method gathers vertical features of layers for different soil properties inside of dictionary"""
        keys = self.soil_data_dict.keys()
        soil_params = {param: self.create_vertical_profile(np.array(self.soil_data_dict[param])) for param in keys}
        return soil_params

