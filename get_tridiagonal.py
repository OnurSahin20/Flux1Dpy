import numpy as np 



class Diagonal:
    def __init__(self,z:np.ndarray,soil_model:object,pond_max:float):
        self.soil_model = soil_model
        self.pond = pond_max
        self.N = self.soil_model.ks.shape[0]
        self.ha = -15000
        self.z = z
        self.dz_int = self.z[1:] - self.z[:-1]  
              
        self.dz_node = np.zeros(self.N)
     
        self.dz_node[1:-1] = 0.5 * (self.z[2:] - self.z[:-2]) 
   
        self.dz_node[0] = 0.5 * self.dz_int[0]
        self.dz_node[-1] = 0.5 * self.dz_int[-1]
        

    def create_diagonal(self,hold:np.ndarray,hnew:np.ndarray,dt:float,flux:float,sink:float):
        conduc = self.soil_model.calculate_conductivity(hnew)
        dh = hnew * np.power(10, -3)
        sm_old,sm_new = self.soil_model.calculate_soil_moisture(hold),self.soil_model.calculate_soil_moisture(hnew)
        f1 = self.soil_model.calculate_soil_moisture(hnew + dh)
        capacity = (f1 - sm_old) / dh
        capacity[hnew>=0] = 0
        li = self.N-1
        a,b,c,F = np.zeros(self.N),np.zeros(self.N),np.zeros(self.N),np.zeros(self.N)
  

   
        
        K_mid = 0.5 * (conduc[:-1] + conduc[1:])
        
        # 2. INTERIOR NODES (i = 1 to N-2)

        a[1:-1] = -K_mid[:-1] / self.dz_int[:-1]
        c[1:-1] = -K_mid[1:] / self.dz_int[1:]
        
        b[1:-1] = -a[1:-1] - c[1:-1] + (C[1:-1] * self.dz_node[1:-1] / dt)
        
        F[1:-1] = ( (capacity[1:-1] * self.dz_node[1:-1] / dt) * hnew[1:-1] 
                - ((sm_new[1:-1] - sm_old[1:-1]) * self.dz_node[1:-1] / dt) 
                + (K_mid[1:] - K_mid[:-1]) 
                - (sink[1:-1] * self.dz_node[1:-1]) )


        # TOP BOUNDARY (i = N-1): Neumann (Prescribed Flux)
        if (hnew[-1]<=self.ha):
            # atmospheric boundary handling
            b[li] =1; a[li] = 0; F[li] = 0
        
        elif (hnew[-1]>0 and flux < 0):
            b[li] =1; a[li] = 0; F[li] = 0
        else:
            a[-1] = -K_mid[-1] / self.dz_int[-1]
            c[-1] = 0.0  # No node above
            b[-1] = -a[-1] + (capacity[-1] * self.dz_node[-1] / dt)
            
            F[-1] = ( (capacity[-1] * self.dz_node[-1] / dt) * hnew[-1] 
                    - ((sm_new[-1] - sm_old[-1]) * self.dz_node[-1] / dt) 
                    - flux - K_mid[-1] 
                    - (sink[-1] * self.dz_node[-1]) )

        
        # 4. BOTTOM BOUNDARY (i = 0): Conditional Switch
        
        if self.bottom_bc == "free_drainage":
            # Unit gradient (dh/dz = 0), outflow is strictly gravity-driven (-K)
            a[0] = 0.0
            b[0] = 1.0
            c[0] = -1.0
            F[0] = 0.0
            
        elif self.bottom_bc == "no_flux":
            # Impermeable boundary, no flow crosses z_0
            a[0] = 0.0  # No node below
            c[0] = -K_mid[0] / self.dz_int[0]
            b[0] = -c[0] + (capacity[0] * self.dz_node[0] / dt)
            
            F[0] = ( (capacity[0] * self.dz_node[0] / dt) * hnew[0] 
                - ((sm_new[0] - sm_old[0]) * self.dz_node[0] / dt) 
                + K_mid[0] 
                - (sink[0] * self.dz_node[0]) )
                
        elif self.bottom_bc == "groundwater_table":
            # Dirichlet condition, fixed pressure head
            a[0] = 0.0
            b[0] = 1.0
            c[0] = 0.0
            F[0] = 0
            
        else:
            raise ValueError(f"Unknown bottom boundary flag: {bottom_bc}. "
                            f"Choose 'free_drainage', 'no_flux', or 'groundwater_table'.")

        return a, b, c, F
