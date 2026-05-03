import numpy as np 
from model_builder import InfiltrationModel


pond_max= 3.0
#flux = np.zeros(22, dtype=np.float64)
discretize = {"layers": [49, 51], "dz": [1.0, 1.0]}
flux = np.array([ -6.,-6.,-10,-10,-10,-2.,-2.,0.,0.,0.,1.,1., 1.,1.,1.,1.,1.,1.,1.,1.]) / 1440 
temp_time  = 1440.0
sim_time = temp_time * flux.shape[0]
trans = np.array([0.0] * (flux.shape[0]),dtype=np.float32) # if root wateruptake is active transpiration needs to be given 
bound_bot = 'free drainage'
soil_props = {"tr":[0.065, 0.095],"ths":[0.41, 0.41],"a":[0.075, 0.019],"n": [1.89, 1.31], "m":[1 - (1 / 1.89),1 - (1 / 1.31)],"ks":[106.1 / 1440, 6.24 / 1440], 
"L":[0.5, 0.5]}
hyd_model = 'VGM-AE'
model = InfiltrationModel(sim_time,temp_time,discretize) #create the model!

hini = np.ones(model.nodes,dtype=np.float32) * -100.0
model.set_soil_model(hyd_model,soil_props)

model.set_boundary_conditions(bound_bot)
hout,sout,sink_out= model.set_run_solver(hini,flux,trans,pond_max,time_interval=1)
print(sout)