import jax.numpy as jnp
import jax
from jax.lax import fori_loop

import numpy as np

from diffrax import diffeqsolve, Dopri8, ODETerm, SaveAt, PIDController, CubicInterpolation, backward_hermite_coefficients

from scipy.stats import chi2, norm

from lib.math import init_stat_zeros

from lib.math import mat_lmax_vec, l_max, col_avoid, col_avoid_vmap, UT_col_avoid, UT_col_avoid_vmap, vec_ABAT

import sympy as sp
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

#-----------------
# Gates Error
#-----------------

def gates2Gexek(cur_u, gates, B_k):
    norm_u = jnp.sqrt(cur_u[0]**2 + cur_u[1]**2 + cur_u[2]**2 + 1e-12)
    cov1 = gates[2] ** 2 + (gates[3] * norm_u) ** 2
    cov3 = gates[0] ** 2 + (gates[1] * norm_u) ** 2
    P_exe = jnp.diag(jnp.array([cov1, cov1, cov3]))  # Untransformed Control Covariance

    # Calculate rotation between [0, 0, 1] (third axis parallel to the nominal control vector) and current control vector
    Z_hat = cur_u.flatten() / norm_u
    E_vec = jnp.cross(jnp.array([0., 0., 1.]), Z_hat.flatten())
    E_hat = (E_vec / jnp.sqrt(E_vec[0]**2 + E_vec[1]**2 + E_vec[2]**2 + 1e-12))
    
    S_vec = jnp.cross(E_hat, Z_hat)
    S_hat = S_vec / jnp.sqrt(S_vec[0]**2 + S_vec[1]**2 + S_vec[2]**2 + 1e-12)
    
    rot_mat = jnp.column_stack([S_hat, E_hat, Z_hat])

    G_exe = rot_mat @ jnp.sqrt(P_exe)
    G_exe_k = B_k @ G_exe

    return G_exe_k

def gates2controlnoise(cur_u, gates):
    sig1, sig2, sig3, sig4 = gates

    norm_u = jnp.sqrt(cur_u[0]**2 + cur_u[1]**2 + cur_u[2]**2 + 1e-12)
    cov1 = sig3 ** 2 + (sig4 * norm_u) ** 2
    cov3 = sig1 ** 2 + (sig2 * norm_u) ** 2
    unt_cont_cov = np.diag(np.array([cov1, cov1, cov3]))  # Untransformed Control Covariance

    # Calculate rotation between [0, 0, 1] (third axis parallel to the nominal control vector) and current control vector
    Z_hat = cur_u.flatten() / norm_u
    E_vec = jnp.cross(jnp.array([0., 0., 1.]), Z_hat.flatten())
    E_hat = E_vec / jnp.sqrt(E_vec[0]**2 + E_vec[1]**2 + E_vec[2]**2 + 1e-12)
    
    S_vec = jnp.cross(E_hat, Z_hat)
    S_hat = S_vec / jnp.sqrt(S_vec[0]**2 + S_vec[1]**2 + S_vec[2]**2 + 1e-12)
    
    rot_mat = jnp.column_stack([S_hat, E_hat, Z_hat])

    # Transform control and add appropriate noise
    G_exe_k = rot_mat @ np.sqrt(unt_cont_cov)

    return G_exe_k

#-----------------
# Numerical Integration Subfunctions
#-----------------

def prop_eoms(t, c, args, eoms_eval, Aprop_eval, Bprop_eval):
    us, G_stoch = args

    states = c[:7]
    state_prop = eoms_eval(t, states, us)

    phi_A = c[7:7 + 7 ** 2].reshape(7, 7)
    phi_B = c[7 + 7 ** 2: 7 + 7 ** 2 + 7*3].reshape(7, 3)
    phi_G = c[7 + 7 ** 2 + 7*3: 7 + 7 ** 2 + 7*3 + 7**2].reshape(7, 7)

    A_prop = Aprop_eval(t, states, us)
    B_prop = Bprop_eval(t, states, us)

    A_k_prop = A_prop @ phi_A
    B_k_prop = A_prop @ phi_B + B_prop
    G_k_prop = A_prop@phi_G + phi_G@A_prop.T + G_stoch@G_stoch.T
    out = jnp.hstack((state_prop.flatten(), A_k_prop.flatten(), B_k_prop.flatten(), G_k_prop.flatten()))
    return out.flatten()

def prop_eoms_states(t, c, args, eoms_eval):
    us = args
    state_prop = eoms_eval(t, c, us)
    return state_prop.flatten()

def prop_eoms_mc(t, c, eoms_eval, cur_u, cur_K, error, G_exe, dt, G_stoch):
    states = c[:7]
    us_noise = cur_u.flatten() + (G_exe@norm.rvs(loc=0, scale=1, size=(3, 1))).flatten() + (cur_K@error.reshape(7, 1)).flatten()
    if np.linalg.norm(us_noise) > 1:
        us_noise = us_noise / np.linalg.norm(us_noise)

    state_prop = eoms_eval(t, states, us_noise).flatten() + (G_stoch@norm.rvs(scale=1, size=(7, 1))/np.sqrt(dt)).flatten()
    return state_prop

def propagate_gen(y0, args, int_save, prop):
    term = ODETerm(prop)
    solver = Dopri8()

    t0 = args['t0']
    t1 = args['t1']
    # dt0 = args['dt0']
    us = args['us']
    r_tol =args['r_tol']
    a_tol = args['a_tol']
    G_stoch = args['G_stoch']
    stepsize_controller = PIDController(rtol=r_tol, atol=a_tol)

    save_t = jnp.linspace(t0, t1, int_save)
    sol = diffeqsolve(term, solver, t0, t1, None , y0, args=(us, G_stoch), stepsize_controller=stepsize_controller, saveat=SaveAt(ts=save_t), max_steps=16**3)
    return sol

def propagate_states_gen(y0, args, int_save, prop):
    term = ODETerm(prop)
    solver = Dopri8()

    t0 = args['t0']
    t1 = args['t1']
    # dt0 = args['dt0']
    us = args['us']
    r_tol =args['r_tol']
    a_tol = args['a_tol']
    stepsize_controller = PIDController(rtol=r_tol, atol=a_tol)

    save_t = jnp.linspace(t0, t1, int_save)
    sol = diffeqsolve(term, solver, t0, t1, None, y0, args=us, stepsize_controller=stepsize_controller, saveat=SaveAt(ts=save_t), max_steps=16**3)
    return sol

#-----------------
# (Looped) Numerical Integration
#-----------------
def forward_ode_iteration(i, input_dict, propagate):
    args = input_dict['args']
    us = input_dict['us']
    y0_true_f = input_dict['y0_true_f']

    states = input_dict['states']
    times = input_dict['times']
    A_ks = input_dict['A_ks']
    B_ks = input_dict['B_ks']
    sig_k = input_dict['sig_k']
    t_node_bound = args['t_node_bound']

    args['t0'] = t_node_bound[i]
    args['t1'] = t_node_bound[i + 1]
    args['us'] = us[:, i]

    sol_f = propagate(y0_true_f, args)

    ys = sol_f.ys
    ts = sol_f.ts

    states = states.at[:, :, i].set(ys)
    times = times.at[:, i].set(ts)

    tmp_A = ys[-1, 7:7 + 7 ** 2].reshape(7, 7)
    tmp_B = ys[-1, 7 + 7 ** 2:7 + 7 ** 2 + 7 * 3].reshape(7, 3)
    tmp_int = ys[-1, 7 + 7 ** 2 + 7 * 3:7 + 7 ** 2 + 7 * 3 + 7 ** 2].reshape(7, 7)

    A_ks = A_ks.at[:, :, i].set(tmp_A)
    B_ks = B_ks.at[:, :, i].set(tmp_B)

    sig_k = sig_k.at[:, :, i].set(tmp_int)

    y0_true_f = jnp.hstack((ys[-1, :7].flatten(), jnp.eye(7).flatten(), jnp.zeros(7 * 3 + 7 ** 2)))

    output_dict = {'args': args, 'states': states, 'A_ks': A_ks, 'B_ks': B_ks, 'sig_k': sig_k, 'y0_true_f': y0_true_f, 'us': us, 'times': times}
    return output_dict

def backward_ode_iteration(ii, input_dict, propagate):
    args = input_dict['args']
    us = input_dict['us']
    y0_true_b = input_dict['y0_true_b']

    states = input_dict['states']
    times = input_dict['times']
    A_ks = input_dict['A_ks']
    B_ks = input_dict['B_ks']
    sig_k = input_dict['sig_k']
    t_node_bound = args['t_node_bound']

    backward = args['backward']
    i = backward[ii]

    args['t0'] = t_node_bound[i + 1]
    args['t1'] = t_node_bound[i]
    args['us'] = us[:, i]

    sol_b = propagate(y0_true_b, args)

    ys = sol_b.ys
    ts = sol_b.ts

    states = states.at[:, :, i].set(jnp.flipud(ys))
    times = times.at[:, i].set(jnp.flip(ts))

    tmp_A = ys[-1, 7:7 + 7 ** 2].reshape(7, 7)
    tmp_B = ys[-1, 7 + 7 ** 2:7 + 7 ** 2 + 7 * 3].reshape(7, 3)
    tmp_int = ys[-1, 7 + 7 ** 2 + 7 * 3:7 + 7 ** 2 + 7 * 3 + 7 ** 2].reshape(7, 7)

    A_ks = A_ks.at[:, :, i].set(tmp_A)
    B_ks = B_ks.at[:, :, i].set(tmp_B)

    sig_k = sig_k.at[:, :, i].set(tmp_int)

    y0_true_b = jnp.hstack((ys[-1, :7].flatten(), jnp.eye(7).flatten(), jnp.zeros(7 * 3 + 7 ** 2)))

    output_dict = {'args': args, 'states': states, 'A_ks': A_ks, 'B_ks': B_ks, 'sig_k': sig_k, 'y0_true_b': y0_true_b, 'us': us, 'times': times}
    return output_dict

def forward_ode_iteration_states(i, input_dict, propagate_states):
    args = input_dict['args']
    us = input_dict['us']
    y0_true_f = input_dict['y0_true_f']

    states = input_dict['states']
    times = input_dict['times']
    
    t_node_bound = args['t_node_bound']
    args['t0'] = t_node_bound[i]
    args['t1'] = t_node_bound[i + 1]
    args['us'] = us[:, i]

    sol_f = propagate_states(y0_true_f, args)

    ys = sol_f.ys
    ts = sol_f.ts

    states = states.at[:, :, i].set(ys)
    times = times.at[:, i].set(ts)

    y0_true_f = ys[-1, :7].flatten()

    output_dict = {'args': args, 'states': states, 'y0_true_f': y0_true_f, 'us': us, 'times': times}
    return output_dict

def backward_ode_iteration_states(ii, input_dict,  propagate_states):
    args = input_dict['args']
    us = input_dict['us']
    y0_true_b = input_dict['y0_true_b']

    states = input_dict['states']
    times = input_dict['times']

    backward = args['backward']
    i = backward[ii]

    t_node_bound = args['t_node_bound']
    args['t0'] = t_node_bound[i + 1]
    args['t1'] = t_node_bound[i]
    args['us'] = us[:, i]

    sol_b = propagate_states(y0_true_b, args)

    ys = sol_b.ys
    ts = sol_b.ts

    states = states.at[:, :, i].set(jnp.flipud(ys))
    times = times.at[:, i].set(jnp.flip(ts))

    y0_true_b = ys[-1, :7].flatten()

    output_dict = {'args': args, 'states': states, 'y0_true_b': y0_true_b, 'us': us, 'times': times}
    return output_dict

def forward_cov_iteration(i, input_dict):
    A_ks = input_dict['A_ks']
    B_ks = input_dict['B_ks']
    K_ks = input_dict['K_ks']
    sig_k = input_dict['sig_k']
    P_ks = input_dict['P_ks']
    gates = input_dict['gates']
    xis = input_dict['xis']

    us = input_dict['us']
    backward = input_dict['backward']

    A_k = A_ks[:, :, i]
    sum_k = sig_k[:, :, i]
    B_k = B_ks[:, :, i]

    G_exe_k = gates2Gexek(us[:, i], gates, B_k)

    Bkr = B_k[:3, :3]
    Bkv = B_k[3:6, :3]
    BkrBkrT = Bkr@Bkr.T
    BkvBkvT = Bkv@Bkv.T
    
    u_norm = jnp.sqrt(us[0, i]**2 + us[1, i]**2 + us[2, i]**2 + 1e-12)

    blkdiagr = (xis[0, i])*jnp.linalg.inv(BkrBkrT)
    blkdiagv = (xis[1, i])*jnp.linalg.inv(BkvBkvT)

    weight = jax.scipy.linalg.block_diag(blkdiagr, blkdiagv)
    K_k = -jnp.linalg.inv(jnp.eye(3) + B_k[:6, :].T@weight@B_k[:6, :])@B_k[:6, :].T@weight@A_k[:6, :6]
    K_ks = K_ks.at[:, :, i].set(K_k[:, :6])

    act_K = jnp.zeros((3, 7))
    act_K = act_K.at[:, :6].set(K_k[:, :6])
    mod_A = A_k + B_k @ act_K

    cur_P_k = P_ks[:, :, i]
    next_P_k = mod_A @ cur_P_k @ mod_A.T + sum_k + G_exe_k @ G_exe_k.T
    P_ks = P_ks.at[:, :, i + 1].set(next_P_k)

    output_dict = {'A_ks': A_ks, 'B_ks': B_ks, 'K_ks': K_ks, 'sig_k': sig_k, 'P_ks': P_ks, 'us': us, 'backward': backward, 'gates': gates, 'xis': xis}
    return output_dict

def backward_cov_iteration(i, input_dict):
    A_ks = input_dict['A_ks']
    B_ks = input_dict['B_ks']
    K_ks = input_dict['K_ks']
    sig_k = input_dict['sig_k']
    P_ks = input_dict['P_ks']
    gates = input_dict['gates']
    xis = input_dict['xis']

    backward = input_dict['backward']

    us = input_dict['us']

    A_k = jnp.linalg.inv(A_ks[:, :, i])  # A_k_f
    sum_k = A_k @ sig_k[:, :, i] @ A_k.T
    B_k = -A_k @ B_ks[:, :, i]

    G_exe_k = gates2Gexek(us[:, i], gates, B_k)

    Bkr = B_k[:3, :3]
    Bkv = B_k[3:6, :3]
    BkrBkrT = Bkr@Bkr.T
    BkvBkvT = Bkv@Bkv.T
    u_norm = jnp.sqrt(us[0, i]**2 + us[1, i]**2 + us[2, i]**2 + 1e-12)

    blkdiagr = (xis[0, i])*jnp.linalg.inv(BkrBkrT)
    blkdiagv = (xis[1, i])*jnp.linalg.inv(BkvBkvT)

    weight = jax.scipy.linalg.block_diag(blkdiagr, blkdiagv)
    K_k = -jnp.linalg.inv(jnp.eye(3) + B_k[:6, :].T@weight@B_k[:6, :])@B_k[:6, :].T@weight@A_k[:6, :6]
    K_ks = K_ks.at[:, :, i].set(K_k[:, :6])

    act_K = jnp.zeros((3, 7))
    act_K = act_K.at[:, :6].set(K_k[:, :6])
    mod_A = A_k + B_k @ act_K

    cur_P_k = P_ks[:, :, i]
    next_P_k = mod_A @ cur_P_k @ mod_A.T + sum_k + G_exe_k @ G_exe_k.T
    P_ks = P_ks.at[:, :, i + 1].set(next_P_k)

    output_dict = {'A_ks': A_ks, 'B_ks': B_ks, 'K_ks': K_ks, 'sig_k': sig_k, 'P_ks': P_ks, 'us': us, 'backward': backward, 'gates': gates, 'xis': xis}
    return output_dict

#-----------------
# Dynamical Systems
#-----------------

def TwoBodyDynamics(T_max_dim, ve, safe_d):
    r_x, r_y, r_z, v_x, v_y, v_z = sp.symbols('r_x, r_y, r_z, v_x, v_y, v_z', real=True)
    t = sp.symbols("t")
    m = sp.symbols("m", positive=True)
    u1, u2, u3 = sp.symbols('u1, u2, u3', real=True)
    rtmp = sp.symbols('rtmp', positive=True)

    eta = .75
    alpha = 2
    a = (3*(eta*safe_d)**(-3 - alpha))/alpha
    b = (eta*safe_d)**(-3) + a*(eta*safe_d)**alpha

    rval = sp.sqrt(r_x ** 2 + r_y ** 2 + r_z ** 2)
    rmod_term = -a*rval**alpha + b
    rmod = sp.Piecewise((rmod_term, rval <= eta*safe_d), (1/rval**3, rval > eta*safe_d))
    u_norm = sp.sqrt(u1 ** 2 + u2 ** 2 + u3 ** 2 + 1e-12)

    states = sp.Matrix([[r_x],
                        [r_y],
                        [r_z],
                        [v_x],
                        [v_y],
                        [v_z],
                        [m]])

    u1t = u1  # sp.Piecewise((u1, sp.Abs(u1) > 2e-3), (0., sp.Abs(u1) < 2e-3))
    u2t = u2  # sp.Piecewise((u2, sp.Abs(u2) > 2e-3), (0., sp.Abs(u2) < 2e-3))
    u3t = u3  # sp.Piecewise((u3, sp.Abs(u3) > 2e-3), (0., sp.Abs(u3) < 2e-3))
    u_normt = u_norm  # sp.Piecewise((u_norm, sp.Abs(u_norm) > 2e-3), (0., sp.Abs(u_norm) < 2e-3))

    eoms = sp.Matrix([[v_x],
                      [v_y],
                      [v_z],
                      [-r_x * rmod + u1t * T_max_dim / m],
                      [-r_y * rmod + u2t * T_max_dim / m],
                      [-r_z * rmod + u3t * T_max_dim / m],
                      [-u_normt * T_max_dim / ve]])

    inputs = sp.Matrix([[t],
                        [r_x],
                        [r_y],
                        [r_z],
                        [v_x],
                        [v_y],
                        [v_z],
                        [m],
                        [u1],
                        [u2],
                        [u3]])

    controls = sp.Matrix([[u1],
                          [u2],
                          [u3]])

    Aprop = eoms.jacobian(states)
    Bprop = eoms.jacobian(controls)
    Cprop = eoms - Aprop @ states - Bprop @ controls

    eoms_eval = sp.lambdify([t, states, controls], eoms, 'jax')
    Aprop_eval = sp.lambdify([t, states, controls], Aprop, 'jax')
    Bprop_eval = sp.lambdify([t, states, controls], Bprop, 'jax')
    Cprop_eval = sp.lambdify([t, states, controls], Cprop, 'jax')

    return eoms_eval, Aprop_eval, Bprop_eval, Cprop_eval

def CR3BPDynamics(T_max_dim, ve, mu, safe_d):
    r_x, r_y, r_z, v_x, v_y, v_z = sp.symbols('r_x, r_y, r_z, v_x, v_y, v_z', real=True)
    t = sp.symbols("t")
    m = sp.symbols("m", positive=True)
    u1, u2, u3 = sp.symbols('u1, u2, u3', real=True)
    dtmp, rtmp = sp.symbols('dtmp, rtmp', positive=True)

    eta = .75
    alpha = 2
    a = (3*(eta*safe_d)**(-3 - alpha))/alpha
    b = (eta*safe_d)**(-3) + a*(eta*safe_d)**alpha

    dval = sp.sqrt((r_x + mu) ** 2 + r_y ** 2 + r_z ** 2)
    rval = sp.sqrt((r_x - 1 + mu) ** 2 + r_y ** 2 + r_z ** 2)

    u_norm = sp.sqrt(u1 ** 2 + u2 ** 2 + u3 ** 2 + 1e-12)

    states = sp.Matrix([[r_x],
                        [r_y],
                        [r_z],
                        [v_x],
                        [v_y],
                        [v_z],
                        [m]])

    term1 = -(1 - mu)*(r_x + mu)*dtmp - mu*(r_x - 1 + mu)*rtmp + r_x
    term2 = -(1 - mu)*r_y*dtmp - mu*r_y*rtmp + r_y
    term3 = -(1 - mu)*r_z*dtmp - mu*r_z*rtmp

    dmod_term = -a*dval**alpha + b
    rmod_term = -a*rval**alpha + b

    dmod = sp.Piecewise((dmod_term, dval <= eta*safe_d), (1/dval**3, dval > eta*safe_d))
    rmod = sp.Piecewise((rmod_term, rval <= eta*safe_d), (1/rval**3, rval > eta*safe_d))
    subs_dict = {rtmp: rmod, dtmp: dmod}

    eoms_pre = sp.Matrix([[v_x],
                      [v_y],
                      [v_z],
                      [term1 + 2 * v_y + u1 * T_max_dim / m],
                      [term2 - 2 * v_x + u2 * T_max_dim / m],
                      [term3 + u3 * T_max_dim / m],
                      [-u_norm * T_max_dim / ve]])
    
    eoms = eoms_pre.subs(subs_dict)

    inputs = sp.Matrix([[t],
                        [r_x],
                        [r_y],
                        [r_z],
                        [v_x],
                        [v_y],
                        [v_z],
                        [m],
                        [u1],
                        [u2],
                        [u3]])

    controls = sp.Matrix([[u1],
                          [u2],
                          [u3]])

    Aprop = eoms.jacobian(states)
    Bprop = eoms.jacobian(controls)
    Cprop = eoms - Aprop @ states - Bprop @ controls

    eoms_eval = sp.lambdify([t, states, controls], eoms, 'jax')
    Aprop_eval = sp.lambdify([t, states, controls], Aprop, 'jax')
    Bprop_eval = sp.lambdify([t, states, controls], Bprop, 'jax')
    Cprop_eval = sp.lambdify([t, states, controls], Cprop, 'jax')

    return eoms_eval, Aprop_eval, Bprop_eval, Cprop_eval

#-----------------
# Constraint/Objective Super-Function
#-----------------
def all_constraints(inputs, args, nodes, int_save, for_prop, back_prop):
    output_dict = {}

    # Unpack args
    t_node_bound = args['t_node_bound']
    init_cov = args['init_cov']
    inv_targ_cov_sqrt = args['inv_targ_cov_sqrt']
    y_start = args['y_start']
    y_end = args['y_end']

    r_tol = args['r_tol']
    a_tol = args['a_tol']
    int_save = args['int_save']

    gates = args['gates']
    G_stoch = args['G_stoch']
    eps = args['eps']
    mx = args['mx']
    nodes = args['nodes']

    r_obs = args['r_obs']
    d_safe = args['d_safe']

    det_col_avoid_bool = args['det_col_avoid_bool']
    stat_col_avoid_bool = args['stat_col_avoid_bool']

    alpha_UT = args['alpha_UT']
    beta_UT = args['beta_UT']
    kappa_UT = args['kappa_UT']

    free_phase = args['free_phasing']

    # Unpack inputs
    y0 = inputs['y0']
    yf = inputs['y1']
    us = inputs['us'].reshape(3, nodes)

    xis = inputs['xis'].reshape(2, nodes)

    if free_phase:
        alpha = inputs['alpha']
        beta = inputs['beta']

    # Get node number info
    nodes = np.array(us.shape[1])
    forward = np.arange(0, nodes // 2)
    backward = np.flip(np.arange(nodes // 2, nodes))

    # Initialization

    states, times, A_ks, B_ks, K_ks, sig_k, P_ks = init_stat_zeros(nodes, int_save)
    P_ks = P_ks.at[:, :, 0].set(init_cov)

    # Propagate dynamics
    y0_true_f = jnp.hstack((y0, jnp.eye(7).flatten(), jnp.zeros(7 * 3 + 7 ** 2)))

    args = {'dt0': None, 'r_tol': r_tol, 'a_tol': a_tol, 'G_stoch': G_stoch, 't0': t_node_bound[0], 't1': t_node_bound[0], 'us': us[:, 0], 'forward': forward, 'backward': backward, 'int_save': int_save, 't_node_bound': t_node_bound}

    forward_input_dict = {'args': args, 'states': states, 'A_ks': A_ks, 'B_ks': B_ks, 'sig_k': sig_k, 'y0_true_f': y0_true_f, 'us': us, 'times': times}
    forward_out = fori_loop(forward[0], forward[-1]+1, for_prop, forward_input_dict)

    args = forward_out['args']
    us = forward_out['us']

    states = forward_out['states']
    times = forward_out['times']
    A_ks = forward_out['A_ks']
    B_ks = forward_out['B_ks']
    sig_k = forward_out['sig_k']

    # Propagate backward dynamics
    args['dt0'] =  None
    y0_true_b = jnp.hstack((yf, jnp.eye(7).flatten(), jnp.zeros(7 * 3 + 7 ** 2)))

    backward_input_dict = {'args': args, 'states': states, 'A_ks': A_ks, 'B_ks': B_ks, 'sig_k': sig_k, 'y0_true_b': y0_true_b, 'us': us, 'times': times}

    backward_out = fori_loop(0, len(backward), back_prop, backward_input_dict)

    states = backward_out['states']
    times = backward_out['times']
    A_ks = backward_out['A_ks']
    B_ks = backward_out['B_ks']
    sig_k = backward_out['sig_k']

    cov_data = {'A_ks': A_ks, 'B_ks': B_ks, 'K_ks': K_ks, 'sig_k': sig_k, 'P_ks': P_ks, 'us': us, 'backward': backward, 'gates': gates, 'xis': xis}
    forward_cov_out = fori_loop(forward[0], forward[-1] + 1, forward_cov_iteration, cov_data)

    A_ks = forward_cov_out['A_ks']
    B_ks = forward_cov_out['B_ks']
    sig_k = forward_cov_out['sig_k']
    P_ks = forward_cov_out['P_ks']
    K_ks = forward_cov_out['K_ks']

    cov_data = {'A_ks': A_ks, 'B_ks': B_ks, 'K_ks': K_ks, 'sig_k': sig_k, 'P_ks': P_ks, 'us': us, 'backward': backward, 'gates': gates, 'xis': xis}
    backward_cov_out = fori_loop(backward[-1], backward[0]+1, backward_cov_iteration, cov_data)

    A_ks = backward_cov_out['A_ks']
    B_ks = backward_cov_out['B_ks']
    sig_k = backward_cov_out['sig_k']
    P_ks = backward_cov_out['P_ks']
    K_ks = backward_cov_out['K_ks']

    node_states = jnp.zeros((nodes+1, 7))
    node_states = node_states.at[0, :].set(states[0, :7, 0])
    node_states = node_states.at[1:, :].set(states[-1, :7, :].T)

    if det_col_avoid_bool:
        col_vals = col_avoid_vmap(node_states, r_obs, d_safe)
        output_dict['c_det_col_avoid'] = col_vals
        col_avoid_out = jnp.max(output_dict['c_det_col_avoid']).astype(float)
    elif stat_col_avoid_bool:
        ybar, Py, weights_m, weights_c, sigmas = UT_col_avoid_vmap(node_states[:, :3], P_ks[:3, :3, :], r_obs, d_safe, alpha_UT, beta_UT, kappa_UT)
        output_dict['c_stat_col_avoid'] = ybar + jax.scipy.stats.norm.ppf(1 - eps) * jnp.sqrt(Py)
        col_avoid_out = jnp.max(output_dict['c_stat_col_avoid']).astype(float)
    else:
        col_avoid_out = jnp.nan

    output_dict['c_ymp'] = states[-1, :7, forward[-1]] - states[0, :7, backward[-1]] # match point

    if free_phase:
        output_dict['c_y0'] = y0[:7] - jnp.hstack([y_start.evaluate(alpha).flatten(), 1.])  # y0 constraint
        output_dict['c_y1'] = yf[:6] - y_end.evaluate(beta).flatten()  # yf constraint
    else:
        output_dict['c_y0'] = y0[:7] - jnp.hstack([y_start, 1.])  # y0 constraint
        output_dict['c_y1'] = yf[:6] - y_end  # yf constraint

    P_us = jnp.einsum('ial, abl, jbl -> ijl', K_ks, P_ks[:6, :6, :-1], K_ks)

    u_const = jnp.sqrt(us[0, :]**2 + us[1, :]**2 + us[2, :]**2 + 1e-12)

    stat_const = mx*jnp.sqrt(mat_lmax_vec(P_us))

    output_dict['c_us'] = u_const + stat_const # u norm constraint

    pyf = inv_targ_cov_sqrt@P_ks[:, :, -1]@inv_targ_cov_sqrt.T - jnp.eye(7)
    output_dict['c_Pyf'] = jnp.log10(l_max(pyf) + 1)

    full_det_cost = jnp.sum(u_const)
    full_stat_cost = jnp.sum(stat_const)
    output_dict['o_mf'] = full_det_cost + full_stat_cost

    base_str = "J_det: {}, J_stat: {}, x0: {}, x1: {}, x_mp: {}, Px: {}, ccol: {}, max_xi: {}"

    jax.debug.print(base_str, full_det_cost.astype(float),
                    full_stat_cost.astype(float),
                    jnp.linalg.norm(output_dict['c_y0'].astype(float)),
                    jnp.linalg.norm(output_dict['c_y1'].astype(float)),
                    jnp.linalg.norm(output_dict['c_ymp'].astype(float)),
                    jnp.max(output_dict['c_Pyf']).astype(float),
                    col_avoid_out,
                    jnp.max(xis).astype(float))

    return output_dict

def all_constraints_states(inputs, args, nodes, int_save, for_prop, back_prop):
    output_dict = {}

    # Unpack args
    t_node_bound = args['t_node_bound']
    y_start = args['y_start']
    y_end = args['y_end']

    r_tol = args['r_tol']
    a_tol = args['a_tol']
    int_save = args['int_save']

    nodes = args['nodes']
    ve = args['ve']
    T_max_dim = args['T_max_dim']

    r_obs = args['r_obs']
    d_safe = args['d_safe']

    free_phase = args['free_phasing']

    # Unpack inputs
    y0 = inputs['y0']
    yf = inputs['y1']
    us = inputs['us'].reshape(3, nodes)

    if free_phase:
        alpha = inputs['alpha']
        beta = inputs['beta']

    # Get node number info
    nodes = np.array(us.shape[1])
    forward = np.arange(0, nodes // 2)
    backward = np.flip(np.arange(nodes // 2, nodes))

    # Initialization

    states = jnp.zeros((int_save, 7, nodes))
    times = jnp.zeros((int_save, nodes))

    # Propagate dynamics
    y0_true_f = y0

    args = {'dt0': None, 'r_tol': r_tol, 'a_tol': a_tol, 't0': t_node_bound[0], 't1': t_node_bound[0], 'us': us[:, 0], 'forward': forward, 'backward': backward, 'int_save': int_save, 't_node_bound': t_node_bound}

    forward_input_dict = {'args': args, 'states': states, 'y0_true_f': y0_true_f, 'us': us, 'times': times}
    forward_out = fori_loop(forward[0], forward[-1]+1, for_prop, forward_input_dict)

    args = forward_out['args']
    us = forward_out['us']

    states = forward_out['states']
    times = forward_out['times']

    # Propagate backward dynamics
    args['dt0'] =  None
    y0_true_b = yf

    backward_input_dict = {'args': args, 'states': states, 'y0_true_b': y0_true_b, 'us': us, 'times': times}

    backward_out = fori_loop(0, len(backward), back_prop, backward_input_dict)

    states = backward_out['states']
    times = backward_out['times']

    node_states = jnp.zeros((nodes+1, 7))
    node_states = node_states.at[0, :].set(states[0, :7, 0])
    node_states = node_states.at[1:, :].set(states[-1, :7, :].T)

    states_flat = states[:, :7, :].transpose(2, 1, 0).reshape(7, -1).T

    col_vals = col_avoid_vmap(node_states, r_obs, d_safe)
    output_dict['c_det_col_avoid'] = col_vals


    output_dict['c_ymp'] = states[-1, :7, forward[-1]] - states[0, :7, backward[-1]] # match point

    if free_phase:
        output_dict['c_y0'] = y0[:7] - jnp.hstack([y_start.evaluate(alpha).flatten(), 1.])  # y0 constraint
        output_dict['c_y1'] = yf[:6] - y_end.evaluate(beta).flatten()  # yf constraint
    else:
        output_dict['c_y0'] = y0[:7] - jnp.hstack([y_start, 1.]) # y0 constraint
        output_dict['c_y1'] = yf[:6] - y_end # yf constraint

    u_const = jnp.sqrt(us[0, :]**2 + us[1, :]**2 + us[2, :]**2 + 1e-12)

    output_dict['c_us'] = u_const # u norm constraint

    output_dict['o_mf'] = jnp.sum(u_const)
    
    base_str = "J: {}, x0: {}, x1: {}, x_mp: {}, ccol: {}"

    jax.debug.print(base_str, output_dict['o_mf'].astype(float),
                    jnp.linalg.norm(output_dict['c_y0'].astype(float)),
                    jnp.linalg.norm(output_dict['c_y1'].astype(float)),
                    jnp.linalg.norm(output_dict['c_ymp'].astype(float)),
                    jnp.max(output_dict['c_det_col_avoid']).astype(float))

    return output_dict

def all_constraints_data(inputs, args, nodes, int_save, for_prop, back_prop):
    output_dict = {}

    # Unpack args
    t_node_bound = args['t_node_bound']
    init_cov = args['init_cov']
    targ_cov = args['targ_cov']
    inv_targ_cov = args['inv_targ_cov']
    inv_targ_cov_sqrt = args['inv_targ_cov_sqrt']
    y_start = args['y_start']
    y_end = args['y_end']

    r_tol = args['r_tol']
    a_tol = args['a_tol']
    int_save = args['int_save']

    gates = args['gates']
    G_stoch = args['G_stoch']
    eps = args['eps']
    mx = args['mx']
    nodes = args['nodes']
    ve = args['ve']
    T_max_dim = args['T_max_dim']

    r_obs = args['r_obs']
    d_safe = args['d_safe']

    det_col_avoid_bool = args['det_col_avoid_bool']
    stat_col_avoid_bool = args['stat_col_avoid_bool']

    alpha_UT = args['alpha_UT']
    beta_UT = args['beta_UT']
    kappa_UT = args['kappa_UT']

    free_phase = args['free_phasing']

    # Unpack inputs
    y0 = inputs['y0']
    yf = inputs['y1']
    us = inputs['us'].reshape(3, nodes)

    xis = inputs['xis'].reshape(2, nodes)
    if free_phase:
        alpha = inputs['alpha']
        beta = inputs['beta']

    # Get node number info
    nodes = np.array(us.shape[1])
    forward = np.arange(0, nodes // 2)
    backward = np.flip(np.arange(nodes // 2, nodes))

    # Initialization

    states = jnp.zeros((int_save, 7 + 7 ** 2 + 7 * 3 + 7 ** 2, nodes))
    times = jnp.zeros((int_save, nodes))

    A_ks = jnp.zeros((7, 7, nodes))
    B_ks = jnp.zeros((7, 3, nodes))

    K_ks = jnp.zeros((3, 6, nodes))

    sig_k = jnp.zeros((7, 7, nodes))

    P_ks = jnp.zeros((7, 7, nodes + 1))
    P_ks = P_ks.at[:, :, 0].set(init_cov)

    dt = t_node_bound[1] - t_node_bound[0]

    # Propagate dynamics
    y0_true_f = jnp.hstack((y0, jnp.eye(7).flatten(), jnp.zeros(7 * 3 + 7 ** 2)))

    args = {'dt0': None, 'r_tol': r_tol, 'a_tol': a_tol, 'G_stoch': G_stoch, 't0': t_node_bound[0], 't1': t_node_bound[0],
            'us': us[:, 0], 'forward': forward, 'backward': backward, 'int_save': int_save, 't_node_bound': t_node_bound}

    forward_input_dict = {'args': args, 'states': states, 'A_ks': A_ks, 'B_ks': B_ks, 'sig_k': sig_k, 'y0_true_f': y0_true_f, 'us': us, 'times': times}
    forward_out = fori_loop(forward[0], forward[-1] + 1, for_prop, forward_input_dict)

    args = forward_out['args']
    us = forward_out['us']

    states = forward_out['states']
    times = forward_out['times']
    A_ks = forward_out['A_ks']
    B_ks = forward_out['B_ks']
    sig_k = forward_out['sig_k']

    # Propagate backward dynamics
    args['dt0'] =  None
    y0_true_b = jnp.hstack((yf, jnp.eye(7).flatten(), jnp.zeros(7 * 3 + 7 ** 2)))

    backward_input_dict = {'args': args, 'states': states, 'A_ks': A_ks, 'B_ks': B_ks, 'sig_k': sig_k,
                           'y0_true_b': y0_true_b, 'us': us, 'times': times}

    backward_out = fori_loop(0, len(backward), back_prop, backward_input_dict)

    states = backward_out['states']
    times = backward_out['times']
    A_ks = backward_out['A_ks']
    B_ks = backward_out['B_ks']
    sig_k = backward_out['sig_k']

    cov_data = {'A_ks': A_ks, 'B_ks': B_ks, 'K_ks': K_ks, 'sig_k': sig_k, 'P_ks': P_ks, 'us': us, 'backward': backward, 'gates': gates, 'xis': xis}
    forward_cov_out = fori_loop(forward[0], forward[-1] + 1, forward_cov_iteration, cov_data)

    A_ks = forward_cov_out['A_ks']
    B_ks = forward_cov_out['B_ks']
    sig_k = forward_cov_out['sig_k']
    P_ks = forward_cov_out['P_ks']
    K_ks = forward_cov_out['K_ks']

    cov_data = {'A_ks': A_ks, 'B_ks': B_ks, 'K_ks': K_ks, 'sig_k': sig_k, 'P_ks': P_ks, 'us': us, 'backward': backward, 'gates': gates, 'xis': xis}
    backward_cov_out = fori_loop(backward[-1], backward[0]+1, backward_cov_iteration, cov_data)

    A_ks = backward_cov_out['A_ks']
    B_ks = backward_cov_out['B_ks']
    sig_k = backward_cov_out['sig_k']
    P_ks = backward_cov_out['P_ks']

    K_ks = backward_cov_out['K_ks']

    node_states = jnp.zeros((nodes + 1, 7))
    node_states = node_states.at[0, :].set(states[0, :7, 0])
    node_states = node_states.at[1:, :].set(states[-1, :7, :].T)

    states_flat = states[:, :7, :].transpose(2, 1, 0).reshape(7, -1).T

    if stat_col_avoid_bool:
        ybar, Py, weights_m, weights_c, sigmas = UT_col_avoid_vmap(node_states[:, :3], P_ks[:3, :3, :], r_obs, d_safe, alpha_UT, beta_UT, kappa_UT)
        output_dict['c_stat_col_avoid'] = ybar + jax.scipy.stats.norm.ppf(1 - eps) * jnp.sqrt(Py)
        col_avoid_out = jnp.max(output_dict['c_stat_col_avoid']).astype(float)

    elif det_col_avoid_bool:
        col_vals = col_avoid_vmap(node_states, r_obs, d_safe)
        output_dict['c_det_col_avoid'] = col_vals
        col_avoid_out = jnp.max(output_dict['c_det_col_avoid']).astype(float)
    else:
        col_avoid_out = jnp.nan

    output_dict['c_ymp'] = states[-1, :7, forward[-1]] - states[0, :7, backward[-1]]  # match point

    if free_phase:
        output_dict['c_y0'] = y0[:7] - jnp.hstack([y_start.evaluate(alpha).flatten(), 1.])  # y0 constraint
        output_dict['c_y1'] = yf[:6] - y_end.evaluate(beta).flatten()  # yf constraint
    else:
        output_dict['c_y0'] = y0[:7] - jnp.hstack([y_start, 1.])  # y0 constraint
        output_dict['c_y1'] = yf[:6] - y_end  # yf constraint

    P_us = jnp.einsum('ial, abl, jbl -> ijl', K_ks, P_ks[:6, :6, :-1], K_ks)

    u_const = jnp.sqrt(us[0, :]**2 + us[1, :]**2 + us[2, :]**2 + 1e-12)

    stat_const = mx*jnp.sqrt(mat_lmax_vec(P_us))
    stat_costs = mx*jnp.sqrt(mat_lmax_vec(P_us))
    output_dict['c_us'] = u_const + stat_const  # u norm constraint

    pyf = inv_targ_cov_sqrt@P_ks[:, :, -1]@inv_targ_cov_sqrt.T - jnp.eye(7)
    output_dict['c_Pyf'] = jnp.log10(l_max(pyf) + 1)
    full_det_cost = jnp.sum(u_const)
    full_stat_cost = jnp.sum(stat_costs)
    output_dict['o_mf'] = full_det_cost + full_stat_cost

    # base_str = "J_det: {}, J_stat: {}, x0: {}, x1: {}, x_mp: {}, Px: {}"
    #
    # jax.debug.print(base_str, full_det_cost.astype(float),
    #                 full_stat_cost.astype(float),
    #                 jnp.linalg.norm(output_dict['c_y0'].astype(float)),
    #                 jnp.linalg.norm(output_dict['c_y1'].astype(float)),
    #                 jnp.linalg.norm(output_dict['c_ymp'].astype(float)),
    #                 output_dict['c_Pyf'].astype(float))

    all_info_dict = {'states': states, 'times': times, 'A_ks': A_ks, 'B_ks': B_ks, 'sig_k': sig_k, 'P_ks': P_ks, 'P_us': P_us, 'obj_constr': output_dict, 'K_ks': K_ks}

    return all_info_dict

#-----------------
# Misc Propagation
#-----------------

def one_monte_carlo(args, inputs, eoms_eval):
    nodes = args['nodes'] 
    eoms_eval = args['eoms_eval']
    t_node_bound = args['t_node_bound']
    states = args['states']
    ybar = args['ybar']
    gates =  args['gates']
    init_cov =  args['init_cov']
    int_save = args['int_save']
    mc_div = args['mc_div']
    G_stoch = args['G_stoch']

    us = inputs['us'].reshape(3, nodes)

    K_ks = inputs['K_ks'].reshape(3, 6, nodes)

    cur_trial_ys = [0]*nodes
    cur_trial_ts = [0]*nodes
    control_data = [0]*nodes

    dt = (t_node_bound[1] - t_node_bound[0])/int_save/mc_div

    y0 = inputs['y0']
    y0_true = norm.rvs(loc=y0, scale=np.sqrt(init_cov.diagonal()))

    for i in range(nodes):
        t0 = t_node_bound[i]
        tf = t_node_bound[i+1]

        cur_u = us[:, i]
        cur_K = np.zeros((3, 7))
        cur_K[:, :6] = K_ks[:, :, i]

        error = y0_true.flatten() - states[0, :7, i].flatten()
        G_exe = gates2controlnoise(cur_u, gates)

        sol = solve_ivp(prop_eoms_mc, y0=y0_true, t_span=[t0, tf], args=(eoms_eval, cur_u, cur_K, error, G_exe, dt, G_stoch), rtol=1e10, atol=1e10, max_step=dt, method='DOP853')
        cur_trial_ys[i] = sol.y.T
        cur_trial_ts[i] = sol.t

        y0_true = sol.y.T[-1, :7].flatten()
        control_data[i] = np.tile((cur_K @ error).T, (len(sol.t), 1))

    all_ys = np.row_stack(cur_trial_ys)
    all_ts = np.hstack(cur_trial_ts)
    dense_ys = interp1d(all_ts, all_ys, axis=0)

    final_control = np.vstack(control_data)
    dense_us = interp1d(all_ts, final_control, axis=0, kind='previous')

    return dense_ys, dense_us

def propagate_states_perorb(y0, args, prop):
    t0 = args['t0']
    t1 = args['t1']
    r_tol = args['r_tol']
    a_tol = args['a_tol']
    term = ODETerm(prop)
    solver = Dopri8()

    stepsize_controller = PIDController(rtol=r_tol, atol=a_tol)

    sol = diffeqsolve(term, solver, t0, t1, None, jnp.hstack([y0, 1.]), stepsize_controller=stepsize_controller, saveat=SaveAt(ts=jnp.linspace(t0, t1, 1000)), max_steps=16**3, args=jnp.zeros(3))
    return sol.ys, sol.ts