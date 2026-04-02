import numpy as np 
from model_build import InfiltrationModel

pond_max= 3.0
flux = np.zeros(20, dtype=np.float64)
discretize = {"layers": [49, 51], "dz": [1.0, 1.0]}
flux[0:2],flux[2:5],flux[5:7],flux[7:10],flux[10:20] = 0.1,0.1,0.1,0.1,0.1
print(flux)
flux = flux / (1440) 
print(flux)
temp_time  = 1440.0
sim_time = temp_time * flux.shape[0]
flux_bot = np.zeros(flux.shape)
head_bot,head_top = np.zeros(flux.shape),np.ones(flux.shape)
trans = np.array([0.0] * (flux.shape[0]),dtype=np.float64) # if root wateruptake is active transpiration needs to be given 
#bound_bot = 'seepage face'
bound_bot = 'constant head'
bound_top = 'atmospheric'
soil_props = {"tr":[0.065, 0.095],"ths":[0.41, 0.41],"a":[0.075, 0.019],"n": [1.89, 1.31], "m":[1 - (1 / 1.89),1 - (1 / 1.31)],"ks":[106.1 / 1440, 6.24 / 1440], 
"L":[0.5, 0.5]}
hyd_model = 'VGM'
model = InfiltrationModel(sim_time,temp_time,discretize) #create the model!
hini = np.ones(model.nodes,dtype=np.float64) * -5
model.set_soil_model(hyd_model,soil_props)
model.set_boundary_conditions(hini,bound_top,bound_bot,pond_max)
hout = model.set_run_solver(hini,flux,flux_bot,head_top,head_bot,trans)
print(hout[:,-1])