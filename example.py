import numpy as np
from model_build import InfiltrationModel
#input parameters for the model!



# soil properties of Genuchten-VGM model
soil_properties = {"a": [2.8 / 100, 1.04 / 100, 2.8 / 100], "tr": [0.029, 0.106, 0.029],
                   "ths": [0.366, 0.469, 0.366], "ks": [5.41 * 100 / 1440, 0.13 * 100 / 1140, 5.41 * 100 / 1440],
                   "n": [2.239, 1.395, 2.239], "m": [1 - 1 / 2.239, 1 - 1 / 1.395, 1 - 1 / 2.239], "L": [0.5, 0.5, 0.5]}

# soil properties of Genuchten-VGM model

discretize = {"layers": [60, 60, 60], "num_nodes":180, "uniform":True,"dz_min":0.5}
flux = np.array(
    [0.0, 0.0, -0.01, -0.006, -0.008, -0.005, -0.001, -0.003, -0.006, -0.012, -0.012, 0.001, 0.0005, 0.001, 0.0015,
     0.0005, 0.001, 0.001, -0.005, 0.0005, 0.0006]) * 100 / 1440  # unit was converted to centimeter / minute

# flux is net P - Bare evaporation
model = InfiltrationModel(discretize,"VGM",soil_properties,flux,sim_time=1500)

"""
model = InfiltrationModel("day", flux.shape[0], discretize, "VGM", soil_properties, flux, inital)
response_sshape = {"P50":-800,"P0":3}
response_feddes = {"P0": -15, "P0pt": -30, "P2H": -325, "P2L": -600,
                   "P3": -8000}  # feddes response parameters for corn
rwu = RootWaterUptake(model, 150, [0.1] * flux.shape[0], "feddes", response_feddes, "normalized")
model.write_input_files()
rwu.write_rwu_input_files()
"""