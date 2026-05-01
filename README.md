# Flux1Dpy 🌊

**Flux1Dpy** is a fast, lightweight, and fully Python-based 1D unsaturated zone water flow solver. HYDRUS-like subsurface flow simulation capabilities in under 500 lines of code using **Numba**.

## Why use Flux1Dpy?

**Flux1Dpy operates entirely in memory.** Because it requires zero file I/O overhead, for **inverse analysis problems, parameter optimization workflows, and uncertainty estimations.**

## ✨ Key Features
* **High Performance:** Numba for C-like execution speeds.
* **Lightweight:** Clean, modular design with less than 500 lines of code.
* **Flexible Boundary Conditions:** Capable of simulating complex atmospheric forcing (precipitation, evaporation, transpiration) as well as standard boundary conditions.
* **Validated Accuracy:** Tested against HYDRUS-1D, demonstrating highly consistent and reliable results across different simulation examples.

## 🧫 Supported Soil Hydraulic Models
Flux1Dpy includes 5 different soil hydraulic property models, from standard to state-of-the-art:
1. **Brooks and Corey-BC**
2. **Van Genuchten-VGM**
3. **Air-Entry Van Genuchten-VGM-AE**

## 📈 Quick Example
Check jupyter notebook examples from the test folder! ./test/txson_network/field_example.ipynb example has real case data > using meteorological data precipitation, evaporation, transpiration > pedotransfer function derived soil parameters > 1d unsturated zone setting with both custom model and HYDRUS and differences of the models. 5 year simulation more than 40,000 hourly time step real case example.

## 📈 Developing
Current benchmarks show the model is 3x to 5x slower than HYDRUS. Implementing lookup tables for soil hydraulic properties—a feature currently in development—will significantly reduce execution time.


