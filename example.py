import numpy as np
from model_build import InfiltrationModel
from soil_model import SoilModels
#input parameters for the model!

"""
Definition of the arguments in Infiltration Model
length unit is always cm.
time unit is minute!
discrete:dict is the number of layers and length of the layers. Direction is positive upward. [0] index is bottom Example: 
discretize = {"layers": [50, 50], "dz": 1 cm} 50 cm each two layer and dz cell size 1 cm total n = 100 cell
hydraulic_model:str currently supports BC (Brooks and Corey) and VGM (Van-Genuchten), FXW and FXW-M1.
soil_data = {"soil_prop:[p1, p2, p3, ... pl]"} hydraulic properties of the soil model. Example;
soil_properties = {"alpha (1/cm)": [0.075, 0.019],
                   "tr": [0.065, 0.095], "ths": [0.41, 0.41], "Ks (cm/min)": [106.1 / 1440, 6.24 / 1440],
                   "n": [1.89, 1.31], "m": [1-1/1.89,1-1/1.31], "L": [0.5, 0.5]}
flux is net infiltration or exfiltration rate from top of the soil. (precipitation - evaporation). It should be negative
if precipitation bigger than evaporation because direction is positive upward.

rwu_method - Currently S-Shape and Feddes methods are used in rwu. Parameters are the same as defined in the HYDRUS.
rwu_response_paremeters:
for S-Shape there is two input; P50 pressure head corresponding Root water uptake is reduced by 50%
P0, in the root water uptake response function associated with water stress; its recommended value is 3.
Feddes option has five input parameter units are [cm]; P0, roots start to extract water from the soil.
POpt, roots extract water at the maximum possible rate. P2H, roots no longer extract water at the maximum rate for r2H.
    P2L, As above, but for a potential transpiration rate of r2L.
P3, wilting point root water uptake ceases.
rwu_distribution: str is the root distribution bx. Currently "normalized" and "equally" distributed available. If you want
specific root distribution change self.root_distribution parameter
"""


# soil properties of Genuchten-VGM model
soil_props = {"a": [2.8 / 100, 1.04 / 100, 2.8 / 100], "tr": [0.029, 0.106, 0.029],
                   "ths": [0.366, 0.469, 0.366], "ks": [5.41 * 100 / 1440, 0.13 * 100 / 1140, 5.41 * 100 / 1440],
                   "n": [2.239, 1.395, 2.239], "m": [1 - 1 / 2.239, 1 - 1 / 1.395, 1 - 1 / 2.239], "L": [0.5, 0.5, 0.5]}

# soil properties of Genuchten-VGM model
pond_max= 2.5 # > 0 activate the ponding at the top!


discretize = {"layers": [60, 60, 60], "num_nodes":180, "uniform":True,"dz":1}
flux = np.array(
    [0.0, 0.0, -0.01, -0.006, -0.008, -0.005, -0.001, -0.003, -0.006, -0.012, -0.012, 0.001, 0.0005, 0.001, 0.0015,
     0.0005, 0.001, 0.001, -0.005, 0.0005, 0.0006],dtype=np.float64) * 100 / 1440  # unit was converted to centimeter / minute

root_depth = 50 # cm surface to bottom 
#root_distribution = "normalized" # second option is "equally"
root_dist = "equally" # second option is "equally"
plant_func = "feddes" #second option s-shape
feddes_params = response_feddes = {"p0": -10, "p0opt": -25, "p2h": -300, "p2l": -1000,
                   "p3": -8000,"r2h":0.5 / 1440,"r2l":0.1/1440 }  # grass feddes response parameters forcm/days to cm/min conversion of last two parameter
#feddes_params = response_feddes = {"P0": -15, "P0pt": -30, "P2H": -325, "P2L": -600,
 #                  "P3": -8000,"r2H":0.5 / 1440,"r2L":0.1/1440 }  

#response_sshape = {"p50":-800,"p0":3} two parameters are required for sshape
sim_time = 1500 #minute
temp_time  = 60 # temporal resolution of meteorological data 60 hourly 180 3 hourly or 1440 for daily 
transpiration = np.array([0.0] * flux.shape[0],dtype=np.float64) # if root wateruptake is active transpiration needs to be given 
# flux is net P - Bare evaporation
hyd_model = 'VGM'
model = InfiltrationModel(sim_time,temp_time,discretize,hyd_model,soil_props,flux,
                          transpiration,plant_func,feddes_params,root_dist,root_depth,pond_max)

params = model.get_soil_properties()
h = np.array([-10] * params['ks'].shape[0])
soil_class = SoilModels(1,params)
#params = model.get_soil_properties()

#a,n,m,L,tr,ths,ks = soil_properties['a'],soil_properties['n'],soil_properties['m'],soil_properties['L'],soil_properties['tr'],soil_properties['ths'],soil_properties['ks']
