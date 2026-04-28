from numba import njit
import numpy as np


def Numeric_Solver(diagonal,soil_model,root_model,sim_time,temp_time,initial,flux_top,transp,pond_max):
    dt_min = 1/60 # 1 second
    dt_max = temp_time # it can not exceed the temp time 
    dt = 30 # starts from 30 minute !!
    ha = -50_000 # minimum allowed pressure
    hs = pond_max # maximum allowed pressure
    eps_sm = 0.001 # iteration criteria moisture
    eps_h = 1 # iteration criteria pressure head
    @njit(fastmath=True, cache=True)
    def IterateTime(index,pond,hold):
