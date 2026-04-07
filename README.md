# Flux1Dpy 🌊

**Flux1Dpy** is a fast, lightweight, and fully Python-based 1D unsaturated zone water flow solver. Built with Object-Oriented Programming (OOP) concepts and heavily accelerated using **Numba**, it delivers HYDRUS-like simulation capabilities in under 700 lines of code.

## Why use Flux1Dpy?
Traditional soil physics software (like HYDRUS) relies heavily on writing and reading text files and creating temporary folders during simulations. 

**Flux1Dpy operates entirely in memory.** Because it requires zero file I/O overhead, it is incredibly fast and specifically designed to be easily integrated into **inverse analysis problems, parameter optimization workflows, and uncertainty estimations.**

## ✨ Key Features
* **High Performance:** JIT-compiled with Numba for C-like execution speeds.
* **Lightweight:** Clean, modular OOP design with less than 700 lines of code.
* **Flexible Boundary Conditions:** Capable of simulating complex atmospheric forcing (precipitation, evaporation, transpiration) as well as standard head/flux boundary conditions.
* **Validated Accuracy:** Tested against HYDRUS-1D, demonstrating highly consistent and reliable results across different simulation examples.

## 🧫 Supported Soil Hydraulic Models
Flux1Dpy includes 5 different soil hydraulic property models, from standard to state-of-the-art:
1. **Brooks and Corey-BC**
2. **Van Genuchten-VGM**
3. **Air-Entry Van Genuchten-VGM-AE**
4. **(Fredlund-Xing-Wang) FXW** 
5. **FXW-M1**

## 📈 Quick Example
This script sets up a two-layer soil profile, defines atmospheric boundaries, and runs a 20-day infiltration and evaporation simulation.

```python
import numpy as np 
from model_build import InfiltrationModel

# 1. Setup Spatial Discretization and Time
discretize = {"layers": [49, 51], "dz": [1.0, 1.0]}
temp_time = 1440.0 # Time step interval (e.g., minutes per day)

# 2. Define Atmospheric Forcing (Precipitation is negative, Evaporation is positive)
flux = np.array([-6., -6., -10., -10., -10., -2., -2., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]) / 1440 
sim_time = temp_time * flux.shape[0]

# 3. Setup Boundary Arrays
flux_bot = np.zeros(flux.shape)
head_top = np.ones(flux.shape)
head_bot = np.zeros(flux.shape)
trans = np.zeros(flux.shape[0], dtype=np.float64) # Transpiration array

# 4. Define Soil Hydraulic Properties & Boundary Types
bound_bot = 'free drainage'
bound_top = 'atmospheric'
pond_max = 3.0
hyd_model = 'VGM-AE' # Van Genuchten with Air-Entry

soil_props = {"tr": [0.065, 0.095], "ths": [0.41, 0.41],  "a": [0.075, 0.019],  "n": [1.89, 1.31], 
    "m": [1 - (1 / 1.89), 1 - (1 / 1.31)], "ks": [106.1 / 1440, 6.24 / 1440], "L": [0.5, 0.5]}

# 5. Initialize and Run the Model
model = InfiltrationModel(sim_time, temp_time, discretize)
hini = np.ones(model.nodes, dtype=np.float64) * -100 # Initial pressure head

model.set_soil_model(hyd_model, soil_props)
model.set_boundary_conditions(hini, bound_top, bound_bot, pond_max)

# Returns arrays for pressure head, saturation, and root water uptake sink
hout, sout, sink_out = model.set_run_solver(hini, flux, flux_bot, head_top, head_bot, trans)

## 🔮 Future Studies
The solver is actively being developed. Upcoming features and research focus areas include:
* **Real-World Case Studies:** Validating the model's accuracy using data directly from in-situ soil moisture sensors.
* **Inverse Calibration Frameworks:** Building out dedicated tools for parameter optimization and uncertainty estimation.
* **Parallel Computing:** Integrating **Multiprocessing** and **MPI** (Message Passing Interface) capabilities to accelerate massive, large-scale inverse problems and Monte Carlo simulations.

## 💻 Installation & Requirements
Currently, Flux1Dpy requires only standard scientific Python libraries. Clone the repository and ensure you have the following installed:
```bash
pip install numpy numba


