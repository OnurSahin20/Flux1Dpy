from numba import njit
import numpy as np

@njit(cache=True,fastmath=True)
def calculate_error(h1, s1, h2, s2):
    theta_err_max = 0.0
    head_err_max = 0.0
    size = h1.shape[0]
    for i in range(size):
        theta_err = np.abs(s1[i] - s2[i])
        if theta_err > theta_err_max:
            theta_err_max = theta_err
            
        head_err = np.abs(h1[i] - h2[i])
        if head_err > head_err_max:
            head_err_max = head_err     
            
    return theta_err_max, head_err_max

@njit(cache=True,fastmath=True)
def calculate_darcy(h, k, ztop):
    k_mid = (k[-1] + k[-2]) / 2.0
    return -k_mid * ((h[-1] - h[-2]) / ztop + 1.0) 

@njit(cache=True,fastmath=True)
def calculate_pond(qdarcy, pond_old, dt, flux_top, pond_max):
    pond_new = pond_old + (-flux_top + qdarcy) * dt
    if pond_new >= pond_max:
        return pond_max
    if pond_new < 0.0:
        return 0.0
    return pond_new

def Numeric_Solver(diagonal_solver, soil_model,root_model, z, sim_time, temp_time, initial, flux_top, transp, pond_max):
    dt_min = 0.1 # 6 second
    dt_max = temp_time 
    dt_ini = temp_time/10   # Initial dt
    ha = -50_000.0       # Minimum allowed pressure (evaporation limit)
    hs = pond_max        # Maximum allowed pressure
    eps_sm = 0.001 
    eps_h = 1 
    max_iter = 10
    dz_top = np.abs(z[-1] - z[-2])
    
    @njit(cache=True,fastmath=True)
    def IterateTime(dt, pond, hold, current_flux, current_tp):
        success = 1  # 0 = success, 1 = failure
        s_old, k_old, cap_old = soil_model(hold,True) 
        hnew= np.empty(hold.shape, dtype=hold.dtype)
        hnew[:] = hold[:]
        
        if pond > 0.0:
            neumann = 0 
            dirichlet = 1  
            hnew[-1] = pond
        else:
            neumann = 1
            dirichlet = 0
            
        i = 0
        while i < max_iter:
         
            hx = diagonal_solver(dt, s_old, hnew, current_flux, current_tp, neumann)
            
            # Boundary Switching Logic
            if dirichlet == 0:
                if ha < hx[-1] < 0.0:          
                    neumann = 1

                else:
                    neumann = 0
                    dirichlet = 1
                    hnew[:] = hold[:]
                    if hx[-1] >= 0.0:
                        hnew[-1] = 0.0
                    else:
                        hnew[-1] = ha
                        
                    i = 0 
                    continue
            else:
                if hx[-1] >= 0:
                    hnew[-1] = pond 
                else:
                    hnew[-1] = ha
                
            i += 1   
            s1, k1, c1 = soil_model(hnew,True)
            s2, k2, c2 = soil_model(hx,True)        
            err_sm, err_h = calculate_error(hnew, s1, hx, s2)
            
            hnew[:] = hx[:] 
            if (err_sm <= eps_sm) and (err_h <= eps_h):
                 success = 0 # Converged
                 break
        
        # Adaptive Time Stepping Logic
        if i <= 3:
            if dt <= 1.0:
               # Aggressive 100% increase  to escape the tiny-dt trap if it lower than 30 second 
                new_dt = min(dt * 2.0, dt_max) 
            else:
            #Standard 30% increase once we are in a healthy range
                new_dt = min(dt * 1.3, dt_max)
        elif i > 7:
            new_dt = max(dt * 0.8, dt_min)
        else:
            new_dt = dt
        return success, new_dt, hx
    
    @njit(cache=True,fastmath=True)
    def RunSolver(time_interval=1):
        count_time = 0.0
        ind_time = 0.0
        index = 0
        pond = 0.0
        current_dt = dt_ini 
        ini_head = np.empty(initial.shape,initial.dtype)
        ini_head[:] = initial[:]
        r, c = flux_top.shape[0], initial.shape[0]
        out_rows = (r // time_interval) + 1
        hout = np.empty((out_rows, c),initial.dtype)
        sout = np.empty((out_rows, c),initial.dtype)
        sink_out = np.zeros((out_rows, c))
        out_idx = 0
        # Save initial state
        hout[out_idx, :] = ini_head[:]
        s,k,c =  soil_model(ini_head,sm=True)
        sout[out_idx, :] = s[:]
        out_idx += 1

        while count_time < sim_time:
            save_time = temp_time - ind_time
            
            if current_dt > save_time:
                current_dt = save_time
           
            success, new_dt, hx = IterateTime(current_dt, pond, ini_head, flux_top[index], transp[index])
            
            if success != 0: # Failure
                current_dt = current_dt / 3.0
                if current_dt < dt_min:
                    raise ValueError("Solver failed to converge. dt dropped below dt_min")

            else: # Success
                s,k,c = soil_model(hx)
                if hx[-1] >= 0.0: 
                    qdarcy = calculate_darcy(hx, k, dz_top)
                    pond = calculate_pond(qdarcy, pond, current_dt, flux_top[index], hs)
                else:
                    pond = 0.0
               
                count_time += current_dt
                ind_time += current_dt
                current_dt = new_dt 
                ini_head[:] = hx[:] 
                # Output Saving
                if ind_time >= temp_time:
                    if (index + 1) % time_interval == 0:
                        hout[out_idx, :] = ini_head[:]
                        sout[out_idx, :] = s[:]
                        sink = root_model(ini_head[:],transp[index])
                        sink_out[out_idx, :] = sink[:]
                        out_idx += 1

                    ind_time = ind_time - temp_time
                    index += 1
         
        return hout, sout, sink_out
        
    return RunSolver
        
        


