import numpy as np 


class NumericSolver:
    def __init__(self,diagonal:object,ini_head:np.ndarray,sink):
        self.diagonal = diagonal
        self.dt_min = 1/60 #1sec
        self.dt_max = 30 # 30 min
        self.max_count = 25
        self.abs_error = np.power(10,-5)
    