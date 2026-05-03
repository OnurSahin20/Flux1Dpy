import numpy as np
import water_uptake
import create_tridiagonal
import solver
import soil_model

class InfiltrationModel:
    def __init__(self, sim_time: int, temp_time:int,discrete: dict[str:np.ndarray, str:float],precision=np.float32) -> None:
        self.precision = precision
        self.sim_time, self.temp_time = sim_time,temp_time
        self.discrete = discrete
        self.z,self.nodes,self.mat_ids = self.set_grid()
        self.numba_soil,self.numba_root, self.numba_tridia = None, None, None
        self.check  = int(self.sim_time / self.temp_time)
        self.bot_bound = None
        self.soil_params = {}
        self.hydro_model = None
        self.hmin = -1e5
    def validate_components(self):
        for attr in ['numba_soil', 'numba_root', 'numba_tridia']:
            if getattr(self, attr) is None:
                raise ValueError(f"{attr} is None")
            
    def set_grid(self):
        z_list, z = [0.0], 0.0
        mat_ids = [0] 
        c = 0
        for layer_len, dz in zip(self.discrete["layers"], self.discrete["dz"]):
            num_cells = int(round(layer_len / dz)) 
            for _ in range(num_cells):
                z += dz
                z_list.append(z)
                mat_ids.append(c)  # Assign the current material index to the node
            c += 1
        return np.array(z_list, dtype=self.precision), len(z_list), np.array(mat_ids, dtype=np.int32)
    
    def set_run_solver(self,hini,flux_top,trans,pond_max,time_interval=1):
   
        if (self.check != flux_top.shape[0]) or (self.check != trans.shape[0]):
            raise ValueError(f'Simulation time and size of vectors should have to consistent for boundary {self.top_bound}')
        flux_top = flux_top.astype(self.precision)
        trans = trans.astype(self.precision)
        hini = hini.astype(self.precision)
        self.validate_components()
        solv = solver.Numeric_Solver(self.numba_tridia,self.numba_soil,self.numba_root,self.z,self.sim_time,self.temp_time,
                                           hini,flux_top,trans,pond_max,self.hmin,)
        return solv(time_interval)
    
    def create_hgrid(self, h_min=-100000.0, h_trans=-1000.0, total_bins=2500, wet_fraction=0.8):
       
        wet_bins = int(total_bins * wet_fraction)
        dry_bins = total_bins - wet_bins
        
        # 1. DRY ZONE: Logarithmic spacing from h_min to h_trans
        # endpoint=False ensures we don't duplicate the h_trans point
        dry_grid = -np.logspace(np.log10(-h_min), np.log10(-h_trans), dry_bins, endpoint=False)
        
        # 2. WET ZONE: Linear spacing from h_trans to exactly 0.0
        wet_grid = np.linspace(h_trans, 0.0, wet_bins)
        
        # 3. Concatenate and enforce precision
        smart_grid_1d = np.concatenate((dry_grid, wet_grid)).astype(self.precision)
        
        return smart_grid_1d    
    def set_soil_model(self,hydraulic_model,soil_data,h_min=-1e5,use_lut=False,lut_bins=1e4,test=False) -> None:
        self.hydro_model = hydraulic_model
        options = {'BC':0,"VGM":1,'VGM-AE':2}
        num_materials = len(soil_data['ths'])
        self.hmin = h_min
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
                use_lut = False
                self.soil_params[param] = np.full(self.nodes, soil_data[param][0], dtype=self.precision)
            else:
                self.soil_params[param] =  self.create_vertical_profile(soil_data[param])
        if use_lut:
            raw_params = {p: np.array(soil_data[p][:num_materials], dtype=self.precision) for p in req_params}
            self.soil_params = raw_params

        if options[self.hydro_model] == 0:
            self.numba_soil= soil_model.bc_model(self.soil_params['hb'],self.soil_params['ths'],self.soil_params['tr'],self.soil_params['lamb'],self.soil_params['ks'])
        
        elif options[self.hydro_model] == 1:
            self.numba_soil = soil_model.vgm_model(self.soil_params['tr'],self.soil_params['ths'],self.soil_params['ks'],
                                            self.soil_params['a'],self.soil_params['n'],self.soil_params['m'],self.soil_params['L'])
        else:
            self.numba_soil = soil_model.vgm_ae_model(self.soil_params['tr'],self.soil_params['ths'],self.soil_params['ks'],
                                            self.soil_params['a'],self.soil_params['n'],self.soil_params['m'],self.soil_params['L'])

        if use_lut:
            h_grid_1d = -np.logspace(np.log10(-h_min), np.log10(1e-4), int(lut_bins) - 1,dtype=self.precision)
            h_grid_1d = np.append(h_grid_1d, 0.0).astype(self.precision)
            h_table = np.tile(h_grid_1d, (num_materials, 1))
            theta_table, K_table = soil_model.generate_lut_arrays(self.numba_soil, h_table)
            self.numba_soil = soil_model.lut_model(h_table, theta_table, K_table,self.mat_ids)

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
        self.bot_bound = bot_opts[bot_bound]
        if self.numba_root is None:
            bx = np.zeros(self.nodes,self.precision)
            self.numba_root = water_uptake.sshape_model(3,-800,bx) # zero sink_source all the time
        self.numba_tridia = create_tridiagonal.Tridiagonal(self.numba_soil,self.numba_root,self.bot_bound,self.z,self.precision)
   

    def create_vertical_profile(self, x: list) -> np.ndarray:
        return np.array(x, dtype=self.precision)[self.mat_ids] # one line of code creates the vertical profile!

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
                    bx[i] = 1.0 / root_depth
        bx[-1] = 0.0 # neglect the root depth in the top cell. no sink source from it.
        bx[0] = 0.0 # also fron the boundary bot
        return bx


    

