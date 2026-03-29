import numpy as np
from model_build import InfiltrationModel

"""
Definition of the arguments in Infiltration Model
length unit is always cm. time unit is always minute
discrete:dict is the number of layers and dz. Direction is positive upward.
hydraulic_model:str currently supports BC (Brooks and Corey) and VGM (Van-Genuchten), relatively new FXW and FXW-M1 models.
soil_data = {"soil_prop:[p1, p2, p3, ... pl]"} hydraulic properties of the soil model. Example;
flux is net infiltration or exfiltration rate from top of the soil. (precipitation - evaporation)
Rookwater uptake -> 1. S-Shape there is two input; P50 and P0. -> 2. Feddes option has five input parameter: p0,POpt,p2h, p2l,r2L.p3, r2l,r2h.
rwu_distribution: str is the root distribution bx. "normalized" and "equally" distributed available.
"""


# soil properties of Genuchten-VGM model
soil_props = {"a": [0.075, 0.019],
                   "tr": [0.065, 0.095],
                   "ths": [0.41, 0.41],
                   "ks": [106.1 / 1440, 6.24 / 1440],
                   "n": [1.89, 1.31],
                   'm': [1 - 1/1.89,1-1/1.31],
                   'L':[0.5,0.5]}

pond_max= 2.5 # > 0 activate the ponding at the top!

discretize = {"layers": [50, 50], "dz": [2,1]}
bound_bot = 'free drainage' #options : 'constant head', 'constant head', 'constant_flux','variable head','variable flux','free drainage','seepage face'
bound_top = 'constant_head' # options 'constant head','constant_flux','variable head','variable flux','atmospheric'

flux = np.array(
    [-6, -6, -10, 0.0,0.0,0.5,0.5]) / 1440 # unit was converted to centimeter / minute

root_depth = 100 # cm surface to bottom 
root_dist = "normalized" # second option is "equally"
root_model = "feddes" #first option s-shape #response_sshape = {"p50":-800,"p0":3} two parameters are required for sshape
feddes_params = response_feddes = {"p0": -10, "p0opt": -25, "p2h": -300, "p2l": -1000,
                   "p3": -8000,"r2h":0.5 / 1440,"r2l":0.1/1440 }  # grass feddes response parameters forcm/days to cm/min conversion of last two parameter

temp_time  = 1440.0 # temporal resolution of meteorological data 60 (hourly) or 1440 for daily. 
sim_time = temp_time * flux.shape[0] #minute
trans = np.array([0.0] * flux.shape[0],dtype=np.float64) # if root wateruptake is active transpiration needs to be given 
hyd_model = 'VGM'

model = InfiltrationModel(sim_time,temp_time,discretize) #create the model!
model.set_soil_model('VGM',soil_props)
