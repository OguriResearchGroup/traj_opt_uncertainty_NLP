import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np

from pyoptsparse.pySNOPT.pySNOPT import SNOPT
from pyoptsparse import Optimization

import time
from tqdm import tqdm

from scipy.interpolate import interp1d
from scipy.stats import chi2, norm

from joblib import Parallel, delayed

# Utilties
from lib.util import yaml_load, save_data
from lib.util import process_config, choose_dynamics, interpret_bcs, create_args, prepare_prop_funcs, prepare_opt_funcs, process_sparsity
from lib.util import plot_traj, plot_det_traj, plot_dist, plot_weights

# Math
from lib.math import global_nondim_2B, global_nondim_CR3BP
from lib.math import mat_lmax_vec

# Dynamics
from lib.dyn import prop_eoms, prop_eoms_states, propagate_gen, propagate_states_gen, one_monte_carlo
from lib.dyn import all_constraints_data

class sf_red:
    def __init__(self, sf_sol):
        self.xStar = sf_sol.xStar.copy()
        self.optInform = sf_sol.optInform.copy()
        self.constraints = sf_sol.constraints.copy()

if __name__ == "__main__":
    # Inputs
    store_hist = False
    hist_state = 'hist_state.sqlite3' if store_hist else None
    hist = 'hist.sqlite3' if store_hist else None
    N_trials = 64

    cov_scale = 10
    
    config_file = r"configurations/JGCD/case1/config.yaml"
    save_file = r"configurations/JGCD/case1/save.pkl"


    # Data Loading
    config = yaml_load(config_file)

    # Nondimensionalization
    if config['dynamics']['type'] == '2BP':
        mu = config['dynamics']['mu']
        l_star = config['dynamics']['l_star']
        nd = global_nondim_2B(mu, l_star, config['engine']['m0'])

    elif config['dynamics']['type'] == 'CR3BP':
        mu = config['dynamics']['mass_rat']
        l_star = config['dynamics']['l_star']
        Gm1m2 = config['dynamics']['mu1'] + config['dynamics']['mu2']
        nd = global_nondim_CR3BP(Gm1m2, l_star, config['engine']['m0'])

    # Read Config File Options
    int_save, read, file, R_TOL, A_TOL, t0, tf, t_node_bound, N, nodes, forward, backward, m0, Isp, u_max, r_1sig, v_1sig, m_1sig, r_1sig_t, v_1sig_t, m_1sig_t, a_err, fixed_mag, prop_mag, fixed_point, prop_point, eps, detdet, stochdet, stochstoch, alpha_UT, beta_UT, kappa_UT, r_obs, d_safe, optOptions, T_max_dim, init_cov, targ_cov, G_stoch, gates, ve = process_config(config, nd)

    # Choose Appropriate Dynamics
    y0, yf, eoms_eval, Aprop_eval, Bprop_eval, Cprop_eval, dyn_safe = choose_dynamics(config, nd, T_max_dim, ve, d_safe, mu)

    # Propagation Functions with specific number of times to save
    prop_eoms_e, prop_eoms_states_e, forward_ode_iteration_e, backward_ode_iteration_e, forward_ode_iteration_states_e, backward_ode_iteration_states_e = prepare_prop_funcs(propagate_gen, propagate_states_gen, prop_eoms, prop_eoms_states, eoms_eval, Aprop_eval, Bprop_eval, int_save)

    # Interpret Boundary Conditions
    if config['boundary_conditions']['type'] == 'free':
        alpha_low, alpha_high, beta_low, beta_high, y0_interp, yf_interp = interpret_bcs(config, t_node_bound, y0, yf, mu, prop_eoms_states_e)
    
    y0_inp = y0_interp if config['boundary_conditions']['type'] == 'free' else y0
    yf_inp = yf_interp if config['boundary_conditions']['type'] == 'free' else yf

    # Create Arguments for Optimization
    args, args_states = create_args(config, t_node_bound, y0_inp, yf_inp, R_TOL, A_TOL, gates, init_cov, targ_cov, G_stoch, eps, ve, T_max_dim, nodes, alpha_UT, beta_UT, kappa_UT, r_obs, d_safe)

    # Create Functions for Optimization
    vals, grad, sens, vals_states, grad_states, sens_states, all_constraints_data_e = prepare_opt_funcs(args,  args_states, nodes, int_save, forward_ode_iteration_e, backward_ode_iteration_e, forward_ode_iteration_states_e, backward_ode_iteration_states_e)

    # Create Sparsity Patterns
    print("Calculating SNOPT (States) Gradient Sparsity")
    grad_state_sparsity = grad_states({'y0': 1*jnp.ones(7), 'y1': 2*jnp.ones(7), 'us': .1*jnp.ones(3*nodes), 'us': .001*jnp.ones(3*nodes), 'alpha': 0.1, 'beta': 0.1})
    state_proc_sparsity = process_sparsity(grad_state_sparsity)

    print("Calculating SNOPT Gradient Sparsity")
    grad_sparsity = grad({'y0': 1*jnp.ones(7), 'y1': 2*jnp.ones(7), 'us': .01*jnp.ones(3*nodes), 'alpha': 0.1, 'beta': 0.1, 'xis': .01*jnp.ones(2*nodes)})
    grad_proc_sparsity = process_sparsity(grad_sparsity)

    # Deterministic Optimal Control Problem
    optprob_states = Optimization("Forward Backward Without Uncertainty", vals_states)
    optprob_states.addVarGroup("us", 3 * nodes, "c", value=.01, lower=-1, upper=1)
    optprob_states.addVarGroup("y0", 7, "c", lower=[-10, -10, -10, -10, -10, -10, 1e-1], upper=[10, 10, 10, 10, 10, 10, 1],  value=jnp.hstack([y0, 1.]))
    optprob_states.addVarGroup("y1", 7, "c", lower=[-10, -10, -10, -10, -10, -10, 1e-1], upper=[10, 10, 10, 10, 10, 10, 1], value=jnp.hstack([yf, .95]))

    if config['boundary_conditions']['type'] == "free":
        optprob_states.addVarGroup("alpha", 1, lower=alpha_low, upper=alpha_high, value=alpha_low)
        optprob_states.addVarGroup("beta", 1, lower=beta_low, upper=beta_high, value=beta_low)

    optprob_states.addConGroup("c_y0", 7, lower=0, upper=0, wrt=['y0', 'alpha'], jac=state_proc_sparsity['c_y0'])
    optprob_states.addConGroup("c_ymp", 7, lower=0, upper=0, wrt=['us', 'y0', 'y1'], jac=state_proc_sparsity['c_ymp'])
    optprob_states.addConGroup("c_y1", 6, lower=0, upper=0, wrt=['y1', 'beta'], jac=state_proc_sparsity['c_y1'])
    optprob_states.addConGroup("c_us", nodes, upper=1, wrt=['us'], jac=state_proc_sparsity['c_us'])

    optprob_states.addObj("o_mf")

    if config['constraints']['deterministic']['det_col_avoid']['bool']:
        optprob_states.addConGroup("c_det_col_avoid", N, upper=0, jac=state_proc_sparsity['c_det_col_avoid'])

    print('SNOPT (States) Starting')
    start_time = time.time()
    opt_states = SNOPT(options=optOptions)
    sf_snopt_sol_states = opt_states(optprob_states, sens=sens_states, timeLimit=None, storeHistory=hist_state)
    print('SNOPT (States) Finished: %s'%(sf_snopt_sol_states.optInform['text']))
    print("Elapsed Time: %.3f" % (time.time() - start_time))

    # Plot Deterministic Trajectory
    noK_xStar = sf_snopt_sol_states.xStar.copy()
    noK_xStar['xis'] = 0.*jnp.ones(2*nodes)
    noK_data = all_constraints_data_e(noK_xStar)

    plot_det_traj(noK_data, nd, config, eoms_eval, sf_snopt_sol_states, t_node_bound, True, r_obs, d_safe, dyn_safe)
    plot_det_traj(noK_data, nd, config, eoms_eval, sf_snopt_sol_states, t_node_bound, False, r_obs, d_safe, dyn_safe)
    det_data = noK_data.copy()

    # Set up initial guess for stochastic problem
    hot_starter = {}
    hot_starter['us'] = sf_snopt_sol_states.xStar['us']
    hot_starter['xis'] = 1e-4*jnp.ones(2*nodes)

    if config['boundary_conditions']['type'] == "free":
        alpha_0 = sf_snopt_sol_states.xStar['alpha']
        beta_0 = sf_snopt_sol_states.xStar['beta']

    dt = noK_data['times'][-1, 0] - noK_data['times'][0, 0]
    dV99_det = np.sum(np.linalg.norm(sf_snopt_sol_states.xStar['us'].reshape(3, -1).T, axis=1) * T_max_dim / noK_data['states'][0, 6, :] * dt) * nd.v_star
    
    # Stochastic Optimal Control Problem
    optprob = Optimization("Forward Backward Shooting under Uncertainty", vals)
    optprob.addVarGroup("us", 3*nodes, "c",  value=hot_starter['us'], lower=-1, upper=1)
    optprob.addVarGroup("y0", 7, "c", lower=[-10, -10, -10, -10, -10, -10, 1e-1], upper=[10, 10, 10, 10, 10, 10, 1], value=jnp.hstack([y0, 1.]))

    optprob.addVarGroup("y1", 7, "c", lower=[-10, -10, -10, -10, -10, -10, 1e-1], upper=[10, 10, 10, 10, 10, 10, 1], value=jnp.hstack([yf, .95]))

    optprob.addVarGroup('xis', 2*nodes, 'c', lower=1e-5, value=hot_starter['xis'], scale=.1)

    if config['boundary_conditions']['type'] == "free":
        optprob.addVarGroup("alpha", 1, lower=alpha_low, upper=alpha_high, value=alpha_0)
        optprob.addVarGroup("beta", 1, lower=beta_low, upper=beta_high, value=beta_0)

    
    optprob.addConGroup("c_y0", 7, lower=0, upper=0, wrt=['y0', 'alpha'], jac=grad_proc_sparsity['c_y0'])
    optprob.addConGroup("c_ymp", 7, lower=0, upper=0, wrt=['us', 'y0', 'y1'], jac=grad_proc_sparsity['c_ymp'])
    optprob.addConGroup("c_y1", 6, lower=0, upper=0, wrt=['y1', 'beta'], jac=grad_proc_sparsity['c_y1'])
    optprob.addConGroup("c_us", nodes,  upper=1, wrt=['us', 'y0', 'y1', 'xis'], jac=grad_proc_sparsity['c_us'])
    optprob.addConGroup("c_Pyf", 1, upper=0., wrt=['us', 'y0', 'y1', 'xis'], jac=grad_proc_sparsity['c_Pyf'])

    if config['constraints']['stochastic']['stat_col_avoid']['bool']:
        optprob.addConGroup("c_stat_col_avoid", N, upper=0, jac=grad_proc_sparsity['c_stat_col_avoid'])

    if config['constraints']['stochastic']['det_col_avoid']['bool']:
        optprob.addConGroup("c_det_col_avoid", N, upper=0, wrt=['us', 'y0', 'y1'], jac=grad_proc_sparsity['c_det_col_avoid'])

    optprob.addObj("o_mf")

    print('SNOPT Starting')
    start_time = time.time()
    opt = SNOPT(options=optOptions)
    sf_snopt_sol = opt(optprob, sens=sens, timeLimit=None, storeHistory=hist)
    print('SNOPT Finished: %s'%(sf_snopt_sol.optInform['text']))
    print("Elapsed Time: %.3f"%(time.time() - start_time))

    # Plot Stochastic Trajectory
    data = all_constraints_data_e(sf_snopt_sol.xStar)
    noK_xStar = sf_snopt_sol.xStar.copy()

    states = np.array(data['states'])
    states_flat = np.hstack(list(states.transpose(2, 1, 0)))

    times = jnp.hstack(data['times'].T)
    times_unique, unique_inds = np.unique(times, return_index=True)
    states_flat_unqiue = states_flat[:7, unique_inds]
    ybar = interp1d(times_unique, states_flat_unqiue)

    # monte_carlo
    eoms_eval_jit = jax.jit(eoms_eval)
    args_mc = {'nodes': nodes, 'eoms_eval': eoms_eval_jit, 't_node_bound': np.asarray(t_node_bound), 'states': states, 'gates': np.asarray(gates), 'init_cov': np.asarray(init_cov), "ybar": ybar, "int_save": config['integration']['int_points'], "mc_div": config['integration']['mc_div'], 'G_stoch': G_stoch}

    inputs = sf_snopt_sol.xStar.copy()
    inputs['K_ks'] = data['K_ks'].flatten()

    # Monte Carlo Trials
    one_monte_carlo_e = lambda args_mc, inputs: one_monte_carlo(args_mc, inputs, eoms_eval)

    mcs = Parallel(n_jobs=8, prefer="threads")(delayed(one_monte_carlo_e)(args_mc, inputs) for i in tqdm(range(N_trials)))

    
    final_mcs = np.zeros((N_trials, 7))
    final_dVs = np.zeros(N_trials)

    for i in range(N_trials):
        cur_mc = mcs[i][0](times[-1]).T - states_flat[:7, -1]

        final_mcs[i, :] = cur_mc * np.array(
            [nd.l_star, nd.l_star, nd.l_star, nd.v_star, nd.v_star, nd.v_star, nd.m_star])

        final_m = mcs[i][0](times[-1]).T[6]
        m99 = 1 - final_m
        final_dVs[i] = ve * np.log(1 / (1 - m99)) * nd.v_star

    us = sf_snopt_sol.xStar['us'].reshape(3, -1).T
    stoch_us = np.sqrt(mat_lmax_vec(data['P_us']))

    dt = data['times'][-1, 0] - data['times'][0, 0]
    detstoch_us = np.linalg.norm(us, axis=1) + chi2.ppf(.99, 3)*stoch_us
    
    # dV99 components
    dV99 = np.sum(detstoch_us*T_max_dim/states[0, 6, :]*dt)*nd.v_star
    dV99_mean = np.sum(np.linalg.norm(us, axis=1)*T_max_dim/states[0, 6, :]*dt)*nd.v_star
    dV99_stat = np.sum(chi2.ppf(.99, 3)*stoch_us*T_max_dim/states[0, 6, :]*dt)*nd.v_star

    # Plotting
    p_traj, det_DF, mc_DF = plot_traj(data, nd, t0, tf, norm.ppf(1-eps), cov_scale, mcs, eoms_eval, sf_snopt_sol, t_node_bound, r_obs, d_safe, config, True, targ_cov, dyn_safe)
    p_traj2, _, _ = plot_traj(data, nd, t0, tf, norm.ppf(1 - eps), cov_scale, mcs, eoms_eval, sf_snopt_sol, t_node_bound, r_obs, d_safe, config, False, targ_cov, dyn_safe)

    plot_dist(nodes, data, states_flat_unqiue, mu, nd, stochdet, stochstoch, mcs, t_node_bound, eps, r_obs, d_safe, alpha_UT, beta_UT, kappa_UT, N_trials, times_unique)

    plot_weights(nodes, data, sf_snopt_sol, nd)

    save_data(sf_snopt_sol_states, sf_snopt_sol, det_data, nd, config, sf_red, t_node_bound, r_obs, d_safe, dyn_safe, data, eps, mcs, t0, tf, targ_cov, ve, mu, dV99_det, dV99, dV99_mean, dV99_stat, T_max_dim, save_file, det_DF, mc_DF)