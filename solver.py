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

def Numeric_Solver(diagonal_solver, soil_model,root_model, z, sim_time, temp_time, initial, flux_top, transp, pond_max,hmin,bot_bound=0):
    dt_min = 0.1 # 6 second
    dt_max = temp_time 
    dt_ini = temp_time/10   # Initial dt
    ha = hmin      # Minimum allowed pressure (evaporation limit)
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
    
    @njit(cache=True, fastmath=True)
    def RunSolver(time_interval=1):
        count_time = 0.0
        ind_time = 0.0
        index = 0
        pond = 0.0
        current_dt = dt_ini 
        
        # State arrays
        ini_head = np.empty(initial.shape, initial.dtype)
        ini_head[:] = initial[:]
        
        # Output arrays
        r, c = flux_top.shape[0], initial.shape[0]
        out_rows = (r // time_interval) + 1
        hout = np.empty((out_rows, c), initial.dtype)
        sout = np.empty((out_rows, c), initial.dtype)
        sink_out = np.zeros((out_rows, c))
        mb_out = np.zeros(out_rows)
        out_idx = 0
        
        # 1. Define exact discrete control volumes (dz)
        dz_vector = np.empty(c, dtype=initial.dtype)
        dz_vector[0] = (z[1] - z[0]) / 2.0
        for i in range(1, c - 1):
            dz_vector[i] = (z[i+1] - z[i-1]) / 2.0
        dz_vector[-1] = (z[-1] - z[-2]) / 2.0
        
        # Save initial state
        hout[out_idx, :] = ini_head[:]
        s, k, cap = soil_model(ini_head, True)
        sout[out_idx, :] = s[:]
        out_idx += 1
        
        # Initialize Mass Balance Trackers (Isolated Soil Column)
        prev_storage = np.sum(s * dz_vector)
        net_flux_sum = 0.0
        abs_flux_sum = 0.0 

        while count_time < sim_time:
            save_time = temp_time - ind_time
            if current_dt > save_time:
                current_dt = save_time
           
            # IterateTime remains untouched, returning only the essential variables
            success, new_dt, hx = IterateTime(current_dt, pond, ini_head, flux_top[index], transp[index])
            
            if success != 0: 
                current_dt = current_dt / 3.0
                if current_dt < dt_min:
                    raise ValueError("Solver failed to converge. Timestep dropped below minimum limit.")
            else: 
                # --- 2. EVALUATE FLUXES POST-CONVERGENCE ---
                s_old, _, _ = soil_model(ini_head, False)
                s_new, k_new, _ = soil_model(hx, False)
                current_sink = root_model(hx, transp[index])
                
                # --- Top Boundary Extraction ---
                if hx[-1] >= 0.0 or hx[-1] <= ha + 1e-5: 
                    # Dirichlet: Reconstruct limited flux balancing the top node
                    k_top_face = (k_new[-1] + k_new[-2]) / 2.0
                    dz_top_face = z[-1] - z[-2]
                    q_top_face = -k_top_face * ((hx[-1] - hx[-2]) / dz_top_face + 1.0) 
                    
                    dS_top = s_new[-1] - s_old[-1]
                    act_q_top = q_top_face - (dS_top * dz_vector[-1]) / current_dt - current_sink[-1] * dz_vector[-1]
                else:
                    # Neumann: Perfectly matches atmospheric flux
                    act_q_top = flux_top[index]
                    

                k_bot_face = (k_new[1] + k_new[0]) / 2.0
                dz_bot_face = z[1] - z[0]
                q_bot_face = -k_bot_face * ((hx[1] - hx[0]) / dz_bot_face + 1.0) 
                
                dS_bot = s_new[0] - s_old[0]
                act_q_bot = q_bot_face + (dS_bot * dz_vector[0]) / current_dt + current_sink[0] * dz_vector[0]
            
                
                # --- Volume Accumulation (Done every dt) ---
                top_vol = -act_q_top * current_dt
                bot_vol = act_q_bot * current_dt
                total_sink_vol = np.sum(current_sink * dz_vector) * current_dt
                
                net_flux_sum += (top_vol + bot_vol - total_sink_vol)
                abs_flux_sum += (np.abs(top_vol) + np.abs(bot_vol) + np.abs(total_sink_vol))
                
                # Update external pond depth
                if hx[-1] >= 0.0: 
                    pond = calculate_pond(act_q_top, pond, current_dt, flux_top[index], hs)
                else:
                    pond = 0.0
               
                # Step Time Forward
                count_time += current_dt
                ind_time += current_dt
                current_dt = new_dt 
                ini_head[:] = hx[:] 
                
                # --- 3. OUTPUT & MASS BALANCE EVALUATION ---
                if ind_time >= temp_time:
                    if (index + 1) % time_interval == 0:
                        hout[out_idx, :] = ini_head[:]
                        sout[out_idx, :] = s_new[:]
                        sink_out[out_idx, :] = current_sink[:]
                        
                        # Calculate final error percentage for the interval
                        current_storage = np.sum(s_new * dz_vector)
                        delta_storage = current_storage - prev_storage
                        mb_error = delta_storage - net_flux_sum
                        
                        if abs_flux_sum > 1e-9:
                            mb_out[out_idx] = (mb_error / abs_flux_sum) * 100.0
                        else:
                            mb_out[out_idx] = 0.0
                            
                        # Reset trackers for the next saving interval
                        prev_storage = current_storage
                        net_flux_sum = 0.0
                        abs_flux_sum = 0.0
                        
                        out_idx += 1

                    ind_time = ind_time - temp_time
                    index += 1
         
        return hout, sout, sink_out, mb_out
        
    return RunSolver
        
        


