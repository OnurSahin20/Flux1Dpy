import numpy as np
from model_build import InfiltrationModel
from soil_model import SoilModels
from source_sink import RootWaterUptake
#input parameters for the model!
from tridiagonal import CreateTriDiagonal
from solver import NumericSolver
"""
Definition of the arguments in Infiltration Model
length unit is always cm.
time unit is always minute
discrete:dict is the number of layers and length of the layers. Direction is positive upward. [0] index is bottom Example: 
hydraulic_model:str currently supports BC (Brooks and Corey) and VGM (Van-Genuchten), relatively new FXW and FXW-M1.
soil_data = {"soil_prop:[p1, p2, p3, ... pl]"} hydraulic properties of the soil model. Example;

flux is net infiltration or exfiltration rate from top of the soil. (precipitation - evaporation). It should be negative
if precipitation bigger than evaporation because direction is positive upward.
for S-Shape there is two input; P50 pressure head corresponding Root water uptake is reduced by 50%
P0, in the root water uptake response function associated with water stress; its recommended value is 3.
Feddes option has five input parameter: p0,POpt,p2h, p2l,r2L.p3, r2l,r2h.
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


discretize = {"layers": [50, 50], "dz": 1,'uniform':True,'num_nodes':100}
#flux = np.array(
    #[-6, -6, -10,  0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1]) / 1440 # unit was converted to centimeter / minute
flux = np.array(
    [-6, -6, -10, 0.0,0.0,0.5,0.5]) / 1440 # unit was converted to centimeter / minute
root_depth = 100 # cm surface to bottom 
#root_distribution = "normalized" # second option is "equally"
root_dist = "normalized" # second option is "equally"
plant_func = "feddes" #second option s-shape
feddes_params = response_feddes = {"p0": -10, "p0opt": -25, "p2h": -300, "p2l": -1000,
                   "p3": -8000,"r2h":0.5 / 1440,"r2l":0.1/1440 }  # grass feddes response parameters forcm/days to cm/min conversion of last two parameter
#feddes_params = response_feddes = {"P0": -15, "P0pt": -30, "P2H": -325, "P2L": -600,
 #                  "P3": -8000,"r2H":0.5 / 1440,"r2L":0.1/1440 }  
tp_mid = 0.3 / 1440
#response_sshape = {"p50":-800,"p0":3} two parameters are required for sshape

temp_time  = 1440.0 # temporal resolution of meteorological data 60 hourly 180 3 hourly or 1440 for daily 
sim_time = temp_time * flux.shape[0]#minute
transpiration = np.array([0.0] * flux.shape[0],dtype=np.float64) # if root wateruptake is active transpiration needs to be given 
# flux is net P - Bare evaporation
hyd_model = 'VGM'
model = InfiltrationModel(sim_time,temp_time,discretize,hyd_model,soil_props,flux,
                          transpiration,plant_func,feddes_params,root_dist,root_depth,pond_max)

params = model.get_soil_properties()
soil_model= SoilModels(1,params) # 1 is VGM and params is dictionary.
root_params  = model.get_root_params()
bx = model.create_root_distribution()
root_model = RootWaterUptake(1,root_params,bx)
diagonal_model= CreateTriDiagonal(soil_model,root_model,1)
hini = np.ones(bx.shape) * -100
numeric_solver = NumericSolver(soil_model,diagonal_model,root_model,sim_time,temp_time,1.0,hini,flux,transpiration,3)

print(numeric_solver.RunSolver())