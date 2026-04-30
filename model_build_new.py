import numpy as np
import soil_model_new
import water_uptake
import create_tridiagonal
import new_solver

class InfiltrationModel:
    def __init__(self, sim_time: int, temp_time:int,discrete: dict[str:np.ndarray, str:float],precision=np.float32) -> None:
        self.sim_time, self.temp_time = sim_time,temp_time
        self.discrete = discrete
        self.z,self.nodes = self.set_grid()
        self.numba_soil,self.numba_root, self.numba_tridia = None, None, None
        self.check  = int(self.sim_time / self.temp_time)
        self.precision = precision
        self.soil_params = {}
        self.hydro_model = None
    def validate_components(self):
        for attr in ['numba_soil', 'numba_root', 'numba_tridia']:
            if getattr(self, attr) is None:
                raise ValueError(f"{attr} is None")
            
    def set_grid(self):
        z_list,z = [0.0], 0.0
        for layer_len, dz in zip(self.discrete["layers"], self.discrete["dz"]):
            num_cells = int(round(layer_len / dz)) 
            for _ in range(num_cells):
                z += dz
                z_list.append(z)    
        return np.array(z_list),len(z_list)  
    
    def set_run_solver(self,hini,flux_top,trans,pond_max,time_interval=1):
   
        if (self.check != flux_top.shape[0]) or (self.check != trans.shape[0]):
            raise ValueError(f'Simulation time and size of vectors should have to consistent for boundary {self.top_bound}')

        self.validate_components()
        solver = new_solver.Numeric_Solver(self.numba_tridia,self.numba_soil,self.z,self.sim_time,self.temp_time,
                                           hini,flux_top,trans,pond_max)
        return solver(time_interval)
        
    def set_soil_model(self,hydraulic_model,soil_data,test=False) -> None:
        self.hydro_model = hydraulic_model
        options = {'BC':0,"VGM":1,'VGM-AE':2}
        if (hydraulic_model== "VGM") or (hydraulic_model== "VGM-AE") :
            req_params = ["a","n","m","tr","ths","ks","L"]
        elif hydraulic_model == "BC":
            req_params = ['lamb','hb',"tr","ths","ks"]
    
        else:
            raise ValueError("Models BC, VGM and VGM-AE are supported!")
        
        if set(soil_data.keys()) != set(req_params):
            raise ValueError(f'Model {hydraulic_model} parameters should be {req_params} but given as {soil_data.keys()}')
        
        for param in soil_data.keys():
            if test:
                self.soil_params[param] =np.array([soil_data[param][0]]*self.z.shape[0]).astype(self.precision)# HYDRUS Profile information for validation
            else:
                self.soil_params[param] =  self.create_vertical_profile(soil_data[param])
       
        if options[self.hydro_model] == 0:
            self.numba_soil= soil_model_new.bc_model(self.soil_params['hb'],self.soil_params['ths'],self.soil_params['tr'],self.soil_params['lamb'],self.soil_params['ks'])
        
        elif options[self.hydro_model] == 1:
            self.numba_soil = soil_model_new.vgm_model(self.soil_params['tr'],self.soil_params['ths'],self.soil_params['ks'],
                                            self.soil_params['a'],self.soil_params['n'],self.soil_params['m'],self.soil_params['L'])
        else:
            self.numba_soil = soil_model_new.vgm_ae_model(self.soil_params['tr'],self.soil_params['ths'],self.soil_params['ks'],
                                            self.soil_params['a'],self.soil_params['n'],self.soil_params['m'],self.soil_params['L'])


    def set_root_model(self,root_model = "",root_params={},
                       root_distribution="", root_depth = 0):
        
        self.bx = self.create_root_distribution(root_distribution,root_depth)
        if root_model == 'feddes':
            self.numba_root = water_uptake.feddes_model(root_params['p0'],root_params['p0opt'],root_params['p2h'],root_params['p2l'],
                                               root_params['p3'],root_params['r2h'],root_params['r2l'],self.bx)
        else:
            self.numba_root = water_uptake.sshape_model(root_params['p0'],root_params['p50'],self.bx)

    
    def set_boundary_conditions(self,bot_bound) -> None:
        bot_opts = {'free drainage':0,'groundwater level':1}
        if self.numba_root is None:
            bx = np.zeros(self.nodes,self.precision)
            self.numba_root = water_uptake.sshape_model(3,-800,bx) # zero sink_source all the time
        self.numba_tridia = create_tridiagonal.Tridiagonal(self.numba_soil,self.numba_root,bot_opts[bot_bound],self.z,self.precision)
   

    def create_vertical_profile(self, x: list) -> np.ndarray:
        vertic_prof = np.zeros(self.nodes,dtype=self.precision)
        c,count = 0,0.0    
        for i in range(self.nodes):
            vertic_prof[i] = x[c]
            if i < self.nodes - 1:
                current_dz = self.discrete["dz"][c]
                count += current_dz
                if c < len(x) - 1 and count >= self.discrete["layers"][c] - 1e-9:
                    c += 1
                    count = 0.0           
        return vertic_prof

    def create_root_distribution(self,root_distribution,root_depth) -> np.ndarray:
        bx = np.zeros(self.z.shape[0], dtype=self.precision)
        z_surf = self.z[-1]
        for i in range(self.z.shape[0]):
            depth = z_surf - self.z[i]
            if i < self.nodes - 1:
                local_dz = self.z[i+1] - self.z[i]
            else:
                local_dz = self.z[i] - self.z[i-1]

            if depth <= root_depth:
                if root_distribution == "normalized":
                    if depth <= 0.2 * root_depth:
                        bx_intensity = 1.667 / root_depth
                    else:
                        bx_intensity = (2.0833 / root_depth) * (1.0 - depth / root_depth)
                    
                    bx[i] = bx_intensity * local_dz
                    
                elif root_distribution == "uniform":
                    bx[i] = 1.0 / self.rzl
        bx[-1] = 0 # neglect the root depth in the top cell. no sink source from it.
        bx[0] = 0 # also fron the boundary bot
        return bx


    

