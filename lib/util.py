import pickle
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.use("webagg")

from jax import numpy as jnp
import jax
from diffrax import diffeqsolve, Dopri8, ODETerm, SaveAt, PIDController, backward_hermite_coefficients, CubicInterpolation

from lib.math import LDL_sqrt, A2q, prop_eoms_util, calc_orbit, sig2cov, calc_t_elapsed_nd, mat_sqrt, UT_col_avoid_vmap
from lib.dyn import TwoBodyDynamics, CR3BPDynamics
from lib.dyn import propagate_states_perorb, forward_ode_iteration_states, backward_ode_iteration_states, forward_ode_iteration, backward_ode_iteration, all_constraints, all_constraints_states, all_constraints_data

from scipy.stats import chi2
from scipy.interpolate import interp1d
from scipy.stats import norm


#-----------------
# IO
#-----------------

def pickle_save(obj, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle)

def pickle_load(filename):
    with open(filename, 'rb') as inp:
        hot_starter = pickle.load(inp)
        return hot_starter
def yaml_load(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config

def yaml_save(config, filename):
    with open(filename, 'w') as file:
        yaml.safe_dump(config, file)
    return

def save_data(sf_snopt_sol_states, sf_snopt_sol, det_data, nd, config, sf_red, t_node_bound, r_obs, d_safe, dyn_safe, data, eps, mcs, t0, tf, targ_cov, ve, mu, dV99_det, dV99, dV99_mean, dV99_stat, T_max_dim, save_file, det_DF, mc_DF):
    print('Saving All Data')
    data_dict = {}

    data_dict['det'] = {'det_data': det_data, 'nd': nd, 'config': config, 'sf_snopt_sol_states': sf_red(sf_snopt_sol_states), 't_node_bound': t_node_bound, 'r_obs': r_obs, 'd_safe': d_safe, 'dyn_safe': dyn_safe, 'dV': dV99_det, 'T_max_dim': T_max_dim, 've': ve, 'mu': mu}

    # stoch -> data, nd, t0, tf, norm.ppf(1-eps), cov_scale, mcs, eoms_eval, sf_snopt_sol, t_node_bound, r_obs, d_safe, config, True, targ_cov, dyn_safe

    data_dict['stoch'] = {'data': data, 'nd': nd, 't0': t0, 'tf': tf, 'mx_norm': norm.ppf(1-eps), 'mcs': mcs, 'sf_snopt_sol': sf_red(sf_snopt_sol), 't_node_bound': t_node_bound, 'r_obs': r_obs, 'd_safe': d_safe, 'config': config, 'targ_cov': targ_cov, 'dyn_safe': dyn_safe, 'dV99': dV99, 'dV_mean': dV99_mean, 'dV_stat': dV99_stat, 'T_max_dim': T_max_dim, 've': ve, 'mu': mu, 'targ_cov': targ_cov, 'det_DF': det_DF, 'mc_DF': mc_DF}

    pickle_save(data_dict, save_file)

#-----------------
# Plotting
#-----------------

def ms(x, y, z, radius, resolution=20):
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)

def get_cov_ellipsoid(cov, mu=np.zeros((3)), nstd=3):
    """
    Return the 3d points representing the covariance matrix
    cov centred at mu and scaled by the factor nstd.

    Plot on your favourite 3d axis.
    Example 1:  ax.plot_wireframe(X,Y,Z,alpha=0.1)
    Example 2:  ax.plot_surface(X,Y,Z,alpha=0.1)

    COPIED FROM https://github.com/CircusMonkey/covariance-ellipsoid/blob/master/ellipsoid.py
    """
    assert cov.shape == (3, 3)

    # Find and sort eigenvalues to correspond to the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.sum(cov, axis=0).argsort()
    eigvals_temp = eigvals[idx]
    idx = eigvals_temp.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Set of all spherical angles to draw our ellipsoid
    n_points = 20
    theta = np.linspace(0, 2 * np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)

    # Width, height and depth of ellipsoid
    rx, ry, rz = nstd * np.sqrt(eigvals)

    # Get the xyz points for plotting
    # Cartesian coordinates that correspond to the spherical angles:
    X = rx * np.outer(np.cos(theta), np.sin(phi))
    Y = ry * np.outer(np.sin(theta), np.sin(phi))
    Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

    # Rotate ellipsoid for off axis alignment
    old_shape = X.shape
    # Flatten to vectorise rotation
    X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
    X, Y, Z = np.matmul(eigvecs, np.array([X, Y, Z]))
    X, Y, Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)

    # Add in offsets for the mean
    X = X + mu[0]
    Y = Y + mu[1]
    Z = Z + mu[2]

    return X, Y, Z

def plot_traj(data, nd, t0, tf, mx, cov_scale, mcs, eoms_eval, sf_sol, t_node_bounds, r_obs, safe_d, config, dark_mode, targ_cov, dyn_safe):
    style = 'dark_background' if dark_mode else 'default'
    plt.style.use(style)
    fig = plt.figure(layout='constrained')
    gs = GridSpec(3, 3, figure=fig)

    col1 = 'lightblue' if dark_mode else 'darkblue'
    col2 = 'white' if dark_mode else 'black'
    col3 = 'yellow' if dark_mode else 'goldenrod'
    col4 = 'lightgray' if dark_mode else 'dimgray'

    ax_traj = fig.add_subplot(gs[:2, :2], projection="3d")

    states = np.array(data['states'])
    traj = np.hstack(list(states.transpose(2, 1, 0))).T
    times = data['times'].T.reshape(-1)

    times_unique, unique_inds = np.unique(times, return_index=True)
    traj_unique = traj[unique_inds, :7]

    prop_eoms_spec = lambda t, y, args: prop_eoms_util(t, y, args, eoms_eval)
    term = ODETerm(prop_eoms_spec)
    solver = Dopri8()

    r_tol = 1e-12
    a_tol = 1e-12
    stepsize_controller = PIDController(rtol=r_tol, atol=a_tol)

    if config['dynamics']['type'] == "CR3BP":
        y0 = jnp.hstack([jnp.array(config['boundary_conditions']['y0']), traj[0, 6]])
        yf = jnp.hstack([jnp.array(config['boundary_conditions']['yf']), traj[-1, 6]])
    elif config['dynamics']['type'] == "2BP":
        y0 = jnp.hstack([calc_orbit(config['boundary_conditions']['coe0'], nd), traj[0, 6]])
        yf = jnp.hstack([calc_orbit(config['boundary_conditions']['coe1'], nd), traj[-1, 6]])

    orb1 = diffeqsolve(term, solver, times[0], times[-1], None, y0, stepsize_controller=stepsize_controller,
                      saveat=SaveAt(dense=True), max_steps=100000)

    orb2 = diffeqsolve(term, solver, times[-1], times[0], None, yf, stepsize_controller=stepsize_controller,
                       saveat=SaveAt(dense=True), max_steps=100000)

    orb1_eval = jax.vmap(orb1.evaluate, in_axes=(0))
    orb2_eval = jax.vmap(orb2.evaluate, in_axes=(0))

    orb1_xyz = orb1_eval(jnp.array(times_unique))[:, :3]*nd.l_star
    orb2_xyz = orb2_eval(jnp.array(times_unique))[:, :3]*nd.l_star

    tf_orb1 = config['boundary_conditions']['per1']
    tf_orb2 = config['boundary_conditions']['per2']

    t_per1 = jnp.linspace(0, tf_orb1, 1000)
    t_per2 = jnp.linspace(0, tf_orb2, 1000)

    orb1_xyz_one = orb1_eval(jnp.array(t_per1))[:, :3]*nd.l_star
    orb2_xyz_one = orb2_eval(jnp.array(t_per2))[:, :3]*nd.l_star
    orb1_dxyz_plot = orb1_eval(t_per1)[:, 3:6]*nd.v_star
    orb2_dxyz_plot = orb2_eval(t_per2)[:, 3:6]*nd.v_star

    times_plot = jnp.linspace(times_unique[0], times_unique[-1], len(times_unique)*20)
    orb1_xyz_plot = orb1_eval(times_plot)[:, :3]*nd.l_star
    orb2_xyz_plot = orb2_eval(times_plot)[:, :3]*nd.l_star

    spacer_traj = len(traj_unique)//10
    spacer1 = len(orb1_xyz_plot)//20
    spacer2 = len(orb2_xyz_plot)//20

    ax_traj.plot(traj_unique[:, 0] * nd.l_star, traj_unique[:, 1] * nd.l_star, traj_unique[:, 2] * nd.l_star, label='Nominal Trajectory', color=col2)
    ax_traj.quiver(traj_unique[::spacer_traj, 0]* nd.l_star, traj_unique[::spacer_traj, 1]* nd.l_star, traj_unique[::spacer_traj, 2]* nd.l_star,
                  traj_unique[::spacer_traj, 3]* nd.v_star,
                  traj_unique[::spacer_traj, 4]* nd.v_star,
                  traj_unique[::spacer_traj, 5]* nd.v_star,
                    color=col2, normalize=True, arrow_length_ratio =10000)

    ax_traj.plot(orb1_xyz_plot[:, 0], orb1_xyz_plot[:, 1], orb1_xyz_plot[:, 2], label='Init Orbit', color='red')
    ax_traj.quiver(orb1_xyz_one[::spacer1, 0], orb1_xyz_one[::spacer1, 1], orb1_xyz_one[::spacer1, 2], 
                  orb1_dxyz_plot[::spacer1, 0], 
                  orb1_dxyz_plot[::spacer1, 1], 
                  orb1_dxyz_plot[::spacer1, 2],
                    color='red', normalize=True, arrow_length_ratio =10000)
    
    ax_traj.plot(orb2_xyz_plot[:, 0], orb2_xyz_plot[:, 1], orb2_xyz_plot[:, 2], label='Final Orbit', color=col3)
    ax_traj.quiver(orb2_xyz_one[::spacer2, 0], orb2_xyz_one[::spacer2, 1], orb2_xyz_one[::spacer2, 2],
                  orb2_dxyz_plot[::spacer2, 0],
                  orb2_dxyz_plot[::spacer2, 1],
                  orb2_dxyz_plot[::spacer2, 2],
                    color=col3, normalize=True, arrow_length_ratio =10000)

    for i in range(len(mcs)):
        # spline_mctraj = pv.Spline(mcs[i][0](times_unique)[:, :3] * nd.l_star, len(times))
        # p_traj.add_mesh(mesh=spline_mctraj, render_lines_as_tubes=True, line_width=2, show_scalar_bar=False,
        #                 color="purple", opacity=.1)

        if i == 0:
            ax_traj.plot([0, 0], [0, 0],
                    [0, 0], color='purple', alpha=1, label="Monte Carlo Trial")
            ax_traj.plot(mcs[i][0](times_unique)[:, 0] * nd.l_star, mcs[i][0](times_unique)[:, 1] * nd.l_star, mcs[i][0](times_unique)[:, 2] * nd.l_star, color='purple', alpha=.05)
        else:
            ax_traj.plot(mcs[i][0](times_unique)[:, 0] * nd.l_star, mcs[i][0](times_unique)[:, 1] * nd.l_star,
                    mcs[i][0](times_unique)[:, 2] * nd.l_star, color='purple', alpha=.05)

    ax_traj.scatter(states[0, 0, 0]* nd.l_star, states[0, 1, 0]* nd.l_star, states[0, 2, 0]* nd.l_star, s=.1, color=col2)
    r_cov = data['P_ks'][:3, :3, 0]* nd.l_star**2

    X1, Y1, Z1 = get_cov_ellipsoid(r_cov**cov_scale**2, mu=states[-1, :3, 0] * nd.l_star, nstd=3)

    # mesh =  pv.StructuredGrid(X1, Y1, Z1)
    ax_traj.plot_surface(X1, Y1, Z1, color='green', alpha=.75)

    for i in range(states.shape[2]):
        if i == 0:
            ax_traj.scatter(states[-1, 0, i]* nd.l_star, states[-1, 1, i]* nd.l_star, states[-1, 2, i]* nd.l_star, color=col2, s=.1)
        else:
            ax_traj.scatter(states[-1, 0, i] * nd.l_star, states[-1, 1, i] * nd.l_star, states[-1, 2, i] * nd.l_star, color=col2, s=.1)
        for j in range(states.shape[0] - 1):
            ax_traj.scatter(states[j, 0, i] * nd.l_star, states[j, 1, i] * nd.l_star, states[j, 2, i] * nd.l_star, color=col2, alpha=.3, s=.1)

        r_cov = data['P_ks'][:3, :3, i+1] * nd.l_star ** 2

        X1, Y1, Z1 = get_cov_ellipsoid(r_cov*cov_scale**2, mu=states[-1, :3, i] * nd.l_star, nstd=3)

        # mesh = pv.StructuredGrid(X1, Y1, Z1)
        if i == 0:
            ax_traj.plot_surface(X1, Y1, Z1, color='green', alpha=.75, label='Covariance')
        else:
            ax_traj.plot_surface(X1, Y1, Z1, color='green', alpha=.75)

    # Optional Collision Avoidance Plotting
    if r_obs is not None:
        X, Y, Z = ms(*r_obs, safe_d)
        ax_traj.plot_surface(X * nd.l_star, Y * nd.l_star, Z * nd.l_star, alpha=.5, color='yellow', label='Keep Out')

        X2, Y2, Z2 = ms(*r_obs, .5*dyn_safe)
        ax_traj.plot_surface(X2 * nd.l_star, Y2 * nd.l_star, Z2 * nd.l_star, alpha=.5, color='red', label='Gravity Limited')

    us = sf_sol.xStar['us'].reshape(3, -1).T
    us_full = np.zeros((traj_unique.shape[0], 3))
    for i in range(len(t_node_bounds) - 1):
        t0 = t_node_bounds[i]
        tf = t_node_bounds[i + 1]

        inds = np.logical_and(times_unique >= t0, times_unique < tf)

        us_full[inds, :] = us[i, :]

    det_data = np.hstack([times_unique.reshape(-1, 1), traj_unique[:, :3], us_full, orb1_xyz/nd.l_star, orb2_xyz/nd.l_star])
    det_data_columns = ['time', 'traj_x', 'traj_y', 'traj_z', 'u1', 'u2', 'u3', 'initial_x', 'initial_y', 'initial_z', 'final_x', 'final_y', 'final_z']
    det_DF = pd.DataFrame(det_data, columns=det_data_columns)
    # det_DF.to_csv('data/det_data.csv', index=False)

    if config['dynamics']['type'] == "CR3BP":
        ax_traj.scatter((1 - config['dynamics']['mass_rat'])*nd.l_star, 0, 0, label="Moon", color=col4)

    ax_traj.set_xlabel('x (km)')
    ax_traj.set_ylabel('y (km)')
    ax_traj.set_zlabel('z (km)')

    ax_traj.axis("equal")

    if dark_mode:
        ax_traj.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_traj.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_traj.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax_traj.legend(loc='lower center', bbox_to_anchor=(-.25, 0))
    ax_traj.set_title('Trajectory')

    # 3 sig

    three_sigs = np.row_stack([3 * np.sqrt(data['P_ks'][:, :, i].diagonal()) for i in range(data['P_ks'].shape[2])])
    three_sig_times = np.stack([data['times'][-1][i] for i in range(len(data['times'][0]))])
    three_sig_times = np.insert(three_sig_times, 0, 0.0)

    states = np.array(data['states'])
    states_flat = np.hstack(list(states.transpose(2, 1, 0)))
    times = jnp.hstack(data['times'].T)

    covs = data['P_ks']
    covus = data['P_us']

    times_unique, unique_inds = np.unique(times, return_index=True)
    cov_final = np.zeros((7, 7, len(times_unique)))
    covu_final = np.zeros((3, 3, len(times_unique)))

    Ntrials = len(mcs)

    gs_axs_3sig = gs[:2, 2].subgridspec(6, 1)
    axs_3sig_00 = fig.add_subplot(gs_axs_3sig[0, 0])
    axs_3sig_10 = fig.add_subplot(gs_axs_3sig[1, 0])
    axs_3sig_20 = fig.add_subplot(gs_axs_3sig[2, 0])
    axs_3sig_01 = fig.add_subplot(gs_axs_3sig[3, 0])
    axs_3sig_11 = fig.add_subplot(gs_axs_3sig[4, 0])
    axs_3sig_21 = fig.add_subplot(gs_axs_3sig[5, 0])


    fig12, ax12 = plt.subplots()

    column_names = ['time']

    for i in range(Ntrials):
        cur_mc = mcs[i][0](times_unique).T - states_flat[:7, unique_inds]

        if i == 0:
            axs_3sig_21.plot([0, 0], [0, 0], 'purple', label="Trial")
        axs_3sig_00.plot(times_unique * nd.t_star / 3600 / 24, cur_mc[0, :] * nd.l_star, 'purple', alpha=.3)
        axs_3sig_10.plot(times_unique * nd.t_star / 3600 / 24, cur_mc[1, :] * nd.l_star, 'purple', alpha=.3)
        axs_3sig_20.plot(times_unique * nd.t_star / 3600 / 24, cur_mc[2, :] * nd.l_star, 'purple', alpha=.3)
        axs_3sig_01.plot(times_unique * nd.t_star / 3600 / 24, cur_mc[3, :] * nd.v_star, 'purple', alpha=.3)
        axs_3sig_11.plot(times_unique * nd.t_star / 3600 / 24, cur_mc[4, :] * nd.v_star, 'purple', alpha=.3)
        axs_3sig_21.plot(times_unique * nd.t_star / 3600 / 24, cur_mc[5, :] * nd.v_star, 'purple', alpha=.3)
        ax12.plot(times_unique * nd.t_star / 3600 / 24, cur_mc[6, :] * nd.m_star, 'purple', alpha=.3)

    for i in range(Ntrials):
        column_names.append('rx_' + str(i))
        column_names.append('ry_' + str(i))
        column_names.append('rz_' + str(i))

    column_names.append("axes_x")
    column_names.append("axes_y")
    column_names.append("axes_z")

    column_names.append("q_x")
    column_names.append("q_y")
    column_names.append("q_z")
    column_names.append("q_w")

    for i in range(Ntrials):
        column_names.append('ux_' + str(i))
        column_names.append('uy_' + str(i))
        column_names.append('uz_' + str(i))

    column_names.append("axes_u_x")
    column_names.append("axes_u_y")
    column_names.append("axes_u_z")

    column_names.append("q_u_x")
    column_names.append("q_u_y")
    column_names.append("q_u_z")
    column_names.append("q_u_w")

    for i in range(len(times_unique)):
        cur_t = times_unique[i]
        bottom_t = t_node_bounds[t_node_bounds <= cur_t][-1]
        bottom_ind = np.where(t_node_bounds == bottom_t)[0][0]
        top_ind = bottom_ind + 1

        if top_ind < len(t_node_bounds):
            top_t = t_node_bounds[top_ind]

            bottom_cov = np.array(covs[:, :, bottom_ind])
            top_cov = np.array(covs[:, :, top_ind])

            bottom_covu = np.array(covus[:, :, bottom_ind])
            top_covu = np.array(covus[:, :, top_ind])

            # cov_final[:, :, i] = covs[:, :, top_ind]
            # covu_final[:, :, i] = covus[:, :, top_ind]

            frac_t = (cur_t - bottom_t) / (top_t - bottom_t)

            Lcov = (1 - frac_t) * LDL_sqrt(bottom_cov) + LDL_sqrt(top_cov) * frac_t
            Lcovu = (1 - frac_t) * LDL_sqrt(bottom_covu) + LDL_sqrt(top_covu) * frac_t

            cov_final[:, :, i] = Lcov @ Lcov.T
            covu_final[:, :, i] = Lcovu @ Lcovu.T
        else:
            cov_final[:, :, i] = covs[:, :, -1]
            covu_final[:, :, i] = covus[:, :, -1]

    axes = np.zeros([cov_final.shape[2], 3])
    quats = np.zeros([cov_final.shape[2], 4])
    for i in range(cov_final.shape[2]):
        cur_cov = cov_final[:, :, i].reshape(7, 7)
        r_cov = cur_cov[:3, :3]

        eigvals, rot_mat = np.linalg.eigh(r_cov)

        axes[i, :] = 3 * np.sqrt(eigvals)

        quats[i, :] = A2q(rot_mat)

    axesu = np.zeros([covu_final.shape[2], 3])
    quatsu = np.zeros([covu_final.shape[2], 4])
    for i in range(cov_final.shape[2]):
        cur_covu = covu_final[:, :, i].reshape(3, 3)
        r_covu = cur_covu[:3, :3]

        eigvalsu, rot_matu = np.linalg.eigh(r_covu)

        axesu[i, :] = 3 * np.sqrt(eigvalsu)

        quatsu[i, :] = A2q(rot_matu)

    mc_data = np.hstack([mcs[i][0](times_unique)[:, :3] for i in range(Ntrials)])
    mc_data2 = np.hstack([mcs[i][1](times_unique) for i in range(Ntrials)])
    mc_data_all = np.hstack([times_unique.reshape(-1, 1), mc_data, axes, quats, mc_data2, axesu, quatsu])
    mc_DF = pd.DataFrame(mc_data_all, columns=column_names)
    # mc_DF.to_csv('data/mc_data.csv', index=False)

    axs_3sig_00.plot(three_sig_times * nd.t_star / 3600 / 24, three_sigs[:, 0] * nd.l_star, '--', color=col2)
    axs_3sig_00.plot(three_sig_times * nd.t_star / 3600 / 24, -three_sigs[:, 0] * nd.l_star, '--', color=col2)

    axs_3sig_10.plot(three_sig_times * nd.t_star / 3600 / 24, three_sigs[:, 1] * nd.l_star, '--', color=col2)
    axs_3sig_10.plot(three_sig_times * nd.t_star / 3600 / 24, -three_sigs[:, 1] * nd.l_star, '--', color=col2)

    axs_3sig_20.plot(three_sig_times * nd.t_star / 3600 / 24, three_sigs[:, 2] * nd.l_star, '--', color=col2)
    axs_3sig_20.plot(three_sig_times * nd.t_star / 3600 / 24, -three_sigs[:, 2] * nd.l_star, '--', color=col2)

    axs_3sig_01.plot(three_sig_times * nd.t_star / 3600 / 24, three_sigs[:, 3] * nd.v_star, '--', color=col2)
    axs_3sig_01.plot(three_sig_times * nd.t_star / 3600 / 24, -three_sigs[:, 3] * nd.v_star, '--', color=col2)

    axs_3sig_11.plot(three_sig_times * nd.t_star / 3600 / 24, three_sigs[:, 4] * nd.v_star, '--', color=col2)
    axs_3sig_11.plot(three_sig_times * nd.t_star / 3600 / 24, -three_sigs[:, 4] * nd.v_star, '--', color=col2)

    axs_3sig_21.plot(three_sig_times * nd.t_star / 3600 / 24, three_sigs[:, 5] * nd.v_star, '--', color=col2, label=r'3$\sigma$')
    axs_3sig_21.plot(three_sig_times * nd.t_star / 3600 / 24, -three_sigs[:, 5] * nd.v_star, '--', color=col2)

    # axs_3sig_00.set_xlabel('time (days)')
    axs_3sig_00.set_ylabel(r'$\varepsilon_{r_x}$ (km)')
    axs_3sig_00.set_xticks([])

    # axs_3sig_10.set_xlabel('time (days)')
    axs_3sig_10.set_ylabel(r'$\varepsilon_{r_y}$ (km)')
    axs_3sig_10.set_xticks([])

    # axs_3sig_20.set_xlabel('time (days)')
    axs_3sig_20.set_ylabel(r'$\varepsilon_{r_z}$ (km)')
    axs_3sig_20.set_xticks([])

    # axs_3sig_01.set_xlabel('time (days)')
    axs_3sig_01.set_ylabel(r'$\varepsilon_{v_x}$ (km/s)')
    axs_3sig_01.set_xticks([])

    # axs_3sig_11.set_xlabel('time (days)')
    axs_3sig_11.set_ylabel(r'$\varepsilon_{v_y}$ (km/s)')
    axs_3sig_11.set_xticks([])

    axs_3sig_21.set_xlabel('time (days)')
    axs_3sig_21.set_ylabel(r'$\varepsilon_{v_z}$ (km/s)')

    axs_3sig_21.legend(loc='center right', bbox_to_anchor=(1, 0.5))
    axs_3sig_00.set_title('Monte Carlo Trials')

    ax12.plot(three_sig_times * nd.t_star / 3600 / 24, three_sigs[:, 6] * nd.m_star, '--', color=col2)
    ax12.plot(three_sig_times * nd.t_star / 3600 / 24, -three_sigs[:, 6] * nd.m_star, '--', color=col2)
    ax12.set_xlabel('time (days)')
    ax12.set_ylabel(r'$\varepsilon_{m}$ (kg)')

    # fig2, ax2 = plt.subplots()

    ax_control = fig.add_subplot(gs[2, 0])

    us = sf_sol.xStar['us'].reshape(3, -1)

    us_interper = interp1d(t_node_bounds[:-1], us, kind='previous', bounds_error=False, fill_value='extrapolate')
    us_interp = us_interper(times_unique)
    stoch_throttle = np.zeros(us_interp.shape[1])

    for i in range(len(t_node_bounds) - 1):
        t0 = t_node_bounds[i]
        tf = t_node_bounds[i + 1]

        inds = np.logical_and(times_unique >= t0, times_unique < tf)

        stoch_throttle[inds] = sf_sol.constraints['c_us'].value[i]
    stoch_throttle[-1] = sf_sol.constraints['c_us'].value[-1]

    ax_control.plot(times_unique * nd.t_star / 3600 / 24, np.linalg.norm(us_interp, axis=0), label='Nominal')
    ax_control.plot(times_unique * nd.t_star / 3600 / 24, stoch_throttle, '--', color=col2, label='99.9% Bound')

    for i in range(Ntrials):
        us_mc_time = us_interp + mcs[i][1](times_unique).T
        norms = np.linalg.norm(us_mc_time, axis=0)
        norms[norms > 1] = 1

        if i == 0:
            ax_control.plot([0, 0], [0, 0], 'purple', label='Trial')
            ax_control.plot(times_unique * nd.t_star / 3600 / 24, norms, 'purple', alpha=.1)
        else:
            ax_control.plot(times_unique * nd.t_star / 3600 / 24, norms, 'purple', alpha=.1)

    ax_control.set_xlabel('time (days)')
    ax_control.set_ylabel('control')
    ax_control.legend(loc='lower center', bbox_to_anchor=(.5, 0))
    ax_control.set_title('Control')

    final_mcs = np.zeros((Ntrials, 7))
    # redim = np.array([nd.l_star, nd.l_star, nd.l_star, nd.v_star, nd.v_star, nd.v_star, nd.m_star])

    for i in range(Ntrials):
        cur_mc = mcs[i][0](times[-1]).T - states_flat[:7, -1]
        final_mcs[i, :] = cur_mc

    # Final Dispersion
    # fig3 = plt.figure()
    # ax3 = plt.axes(projection='3d')

    ax_rdisp = fig.add_subplot(gs[2, 1], projection="3d")

    r_cov = covs[:3, :3, -1]
    X1, Y1, Z1 = get_cov_ellipsoid(r_cov * nd.l_star ** 2, mu=np.zeros(3), nstd=3)
    ax_rdisp.plot_surface(X1, Y1, Z1, alpha=.5, color='green')

    ax_rdisp.plot_surface(X1 * 0, Y1 * 0, Z1 * 0, alpha=1, color='green',
                     label=r'final $3 \sigma$ $\mathbf{r}$')
    ax_rdisp.plot_surface(X1 * 0, Y1 * 0, Z1 * 0, alpha=1, color='red',
                     label=r'target $3 \sigma$ $\mathbf{r}$')

    X2, Y2, Z2 = get_cov_ellipsoid(targ_cov[:3, :3] * nd.l_star ** 2, mu=np.zeros(3),
                                   nstd=3)
    ax_rdisp.plot_surface(X2, Y2, Z2, alpha=.1, color='red')
    for i in range(Ntrials):
        ax_rdisp.scatter(final_mcs[i, 0] * nd.l_star, final_mcs[i, 1] * nd.l_star, final_mcs[i, 2] * nd.l_star, color=col2)

    ax_rdisp.set_xlabel(r'$\varepsilon_{r_x}$ (km)')
    ax_rdisp.set_ylabel(r'$\varepsilon_{r_y}$ (km)')
    ax_rdisp.set_zlabel(r'$\varepsilon_{r_z}$ (km)')
    ax_rdisp.legend(loc='lower center', bbox_to_anchor=(.5, 0))
    ax_rdisp.set_title(r'Final $\mathbf{r}$ Dispersion')

    ax_rdisp.axis("equal")

    if dark_mode:
        ax_rdisp.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_rdisp.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_rdisp.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax_vdisp = fig.add_subplot(gs[2, 2], projection="3d")

    v_cov = covs[3:6, 3:6, -1]
    X1V, Y1V, Z1V = get_cov_ellipsoid(v_cov * nd.v_star ** 2, mu=np.zeros(3), nstd=3)
    ax_vdisp.plot_surface(X1V, Y1V, Z1V, alpha=.5, color='green')

    ax_vdisp.plot_surface(X1V * 0, Y1V * 0, Z1V * 0, alpha=1, color='green',
                     label=r'final $3 \sigma$ $\mathbf{v}$')
    ax_vdisp.plot_surface(X1V * 0, Y1V * 0, Z1V * 0, alpha=1, color='red',
                     label=r'target $3 \sigma$ $\mathbf{v}$')

    X2V, Y2V, Z2V = get_cov_ellipsoid(targ_cov[3:6, 3:6] * nd.v_star ** 2, mu=np.zeros(3),
                                      nstd=3)
    ax_vdisp.plot_surface(X2V, Y2V, Z2V, alpha=.1, color='red')
    for i in range(Ntrials):
        ax_vdisp.scatter(final_mcs[i, 3] * nd.v_star, final_mcs[i, 4] * nd.v_star, final_mcs[i, 5] * nd.v_star, color=col2)

    ax_vdisp.set_xlabel(r'$\varepsilon_{v_x}$ (km)')
    ax_vdisp.set_ylabel(r'$\varepsilon_{v_y}$ (km)')
    ax_vdisp.set_zlabel(r'$\varepsilon_{v_z}$ (km)')
    ax_vdisp.legend(loc='lower center', bbox_to_anchor=(.5, 0))
    ax_vdisp.set_title(r'Final $\mathbf{v}$ Dispersion')

    ax_vdisp.axis("equal")

    if dark_mode:
        ax_vdisp.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_vdisp.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_vdisp.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # fig.tight_layout()
    return fig, det_DF, mc_DF

def plot_det_traj(data, nd, config, eoms_eval, sf_sol, t_node_bounds, dark_mode, r_obs, safe_d, dyn_safe):
    style = 'dark_background' if dark_mode else 'default'
    plt.style.use(style)

    col1 = 'lightblue' if dark_mode else 'darkblue'
    col2 = 'white' if dark_mode else 'black'
    col3 = 'yellow' if dark_mode else 'goldenrod'
    col4 = 'lightgray' if dark_mode else 'dimgray'

    fig =  plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    states = np.array(data['states'])
    traj = np.hstack(list(states.transpose(2, 1, 0))).T
    times = data['times'].T.reshape(-1)

    times_unique, unique_inds = np.unique(times, return_index=True)
    traj_unique = traj[unique_inds, :7]

    prop_eoms_spec = lambda t, y, args: prop_eoms_util(t, y, args, eoms_eval)
    term = ODETerm(prop_eoms_spec)
    solver = Dopri8()

    r_tol = 1e-12
    a_tol = 1e-12
    stepsize_controller = PIDController(rtol=r_tol, atol=a_tol)

    y0 = jnp.hstack([jnp.array(config['boundary_conditions']['y0']), traj[0, 6]])
    yf = jnp.hstack([jnp.array(config['boundary_conditions']['yf']), traj[-1, 6]])

    orb1 = diffeqsolve(term, solver, times[0], times[-1], None, y0, stepsize_controller=stepsize_controller,
                      saveat=SaveAt(dense=True), max_steps=100000)

    orb2 = diffeqsolve(term, solver, times[-1], times[0], None, yf, stepsize_controller=stepsize_controller,
                       saveat=SaveAt(dense=True), max_steps=100000)

    orb1_eval = jax.vmap(orb1.evaluate, in_axes=(0))
    orb2_eval = jax.vmap(orb2.evaluate, in_axes=(0))

    orb1_xyz = orb1_eval(jnp.array(times_unique))[:, :3]*nd.l_star
    orb2_xyz = orb2_eval(jnp.array(times_unique))[:, :3]*nd.l_star

    tf_orb1 = config['boundary_conditions']['per1']
    tf_orb2 = config['boundary_conditions']['per2']

    t_per1 = jnp.linspace(0, tf_orb1, 1000)
    t_per2 = jnp.linspace(0, tf_orb2, 1000)

    orb1_xyz_one = orb1_eval(jnp.array(t_per1))[:, :3]*nd.l_star
    orb2_xyz_one = orb2_eval(jnp.array(t_per2))[:, :3]*nd.l_star
    orb1_dxyz_plot = orb1_eval(t_per1)[:, 3:6]*nd.v_star
    orb2_dxyz_plot = orb2_eval(t_per2)[:, 3:6]*nd.v_star

    times_plot = jnp.linspace(times_unique[0], times_unique[-1], len(times_unique)*20)
    orb1_xyz_plot = orb1_eval(times_plot)[:, :3]*nd.l_star
    orb2_xyz_plot = orb2_eval(times_plot)[:, :3]*nd.l_star


    ax.plot(traj_unique[:, 0] * nd.l_star, traj_unique[:, 1] * nd.l_star, traj_unique[:, 2] * nd.l_star, label='Trajectory', color=col1)
    ax.plot(traj_unique[0, 0] * nd.l_star, traj_unique[0, 1] * nd.l_star, traj_unique[0, 2] * nd.l_star, 'o',
            label='Start', color='cyan')
    ax.plot(traj_unique[-1, 0] * nd.l_star, traj_unique[-1, 1] * nd.l_star, traj_unique[-1, 2] * nd.l_star, 'o',
            label='End', color='green')
    
    spacer_traj = len(traj_unique)//10
    spacer1 = len(orb1_xyz_plot)//20
    spacer2 = len(orb2_xyz_plot)//20

    ax.quiver(traj_unique[::spacer_traj, 0]* nd.l_star, traj_unique[::spacer_traj, 1]* nd.l_star, traj_unique[::spacer_traj, 2]* nd.l_star,
                  traj_unique[::spacer_traj, 3]* nd.v_star,
                  traj_unique[::spacer_traj, 4]* nd.v_star,
                  traj_unique[::spacer_traj, 5]* nd.v_star,
                    color=col2, normalize=True, arrow_length_ratio =10000)
    
    
    ax.plot(orb1_xyz_plot[:, 0], orb1_xyz_plot[:, 1], orb1_xyz_plot[:, 2], label='Init Orbit', color='red')
    ax.quiver(orb1_xyz_one[::spacer1, 0], orb1_xyz_one[::spacer1, 1], orb1_xyz_one[::spacer1, 2], 
                  orb1_dxyz_plot[::spacer1, 0], 
                  orb1_dxyz_plot[::spacer1, 1], 
                  orb1_dxyz_plot[::spacer1, 2],
                    color='red', normalize=True, arrow_length_ratio =10000)
    
    ax.plot(orb2_xyz_plot[:, 0], orb2_xyz_plot[:, 1], orb2_xyz_plot[:, 2], label='Final Orbit', color=col3)
    ax.quiver(orb2_xyz_one[::spacer2, 0], orb2_xyz_one[::spacer2, 1], orb2_xyz_one[::spacer2, 2],
                  orb2_dxyz_plot[::spacer2, 0],
                  orb2_dxyz_plot[::spacer2, 1],
                  orb2_dxyz_plot[::spacer2, 2],
                    color=col3, normalize=True, arrow_length_ratio =10000)

    if config['dynamics']['type'] == "CR3BP":
        ax.scatter((1 - config['dynamics']['mass_rat'])*nd.l_star, 0, 0, label="Moon", color=col4)

    if r_obs is not None:
        X, Y, Z = ms(*r_obs, safe_d)
        ax.plot_surface(X * nd.l_star, Y * nd.l_star, Z * nd.l_star, alpha=.5, color='yellow', label='Keep Out')

        X2, Y2, Z2 = ms(*r_obs, .5*dyn_safe)
        ax.plot_surface(X2 * nd.l_star, Y2 * nd.l_star, Z2 * nd.l_star, alpha=.5, color='red', label='Gravity Limited')

    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_zlabel('z (km)')

    ax.axis("equal")
    if dark_mode:
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.legend(loc="upper left")
    ax.set_title('Trajectory')

    times = jnp.hstack(data['times'].T)
    times_unique, unique_inds = np.unique(times, return_index=True)

    us = sf_sol.xStar['us'].reshape(3, -1)

    us_interper = interp1d(t_node_bounds[:-1], us, kind='previous', bounds_error=False, fill_value='extrapolate')
    us_interp = us_interper(times_unique)

    ax2.plot(times_unique * nd.t_star / 3600 / 24, np.linalg.norm(us_interp, axis=0), label='Nominal Control')
    ax2.set_xlabel('time (days)')
    ax2.set_ylabel('control')
    ax2.set_title('Control')
    ax2.legend(loc='upper left')

    return fig

def plot_dist(nodes, data, states_flat_unqiue, mu, nd, stochdet, stochstoch, mcs, t_node_bound, eps, r_obs, d_safe, alpha_UT, beta_UT, kappa_UT, N_trials, times_unique):
    plt.figure()
    node_states = jnp.zeros((nodes+1, 7))
    node_states = node_states.at[0, :].set(data['states'][0, :7, 0])
    node_states = node_states.at[1:, :].set(data['states'][-1, :7, :].T)
    moon_dist = jnp.linalg.norm(states_flat_unqiue[:3, :].T - jnp.array([1 - mu, 0, 0]), axis=1)
    
    if stochdet:
        col = jnp.linalg.norm(node_states[:, :3] - jnp.array([1 - mu, 0, 0]), axis=1)*nd.l_star
        lab2 = "Deterministic Radius"
    elif stochstoch:
        ybar, Py, _, _, _ = UT_col_avoid_vmap(node_states[:, :3], data['P_ks'][:3, :3, :], r_obs, d_safe, alpha_UT, beta_UT, kappa_UT)
        col = (jnp.linalg.norm(node_states[:, :3] - jnp.array([1 - mu, 0, 0]), axis=1) - jax.scipy.stats.norm.ppf(1 - eps) *jnp.sqrt(Py))*nd.l_star
        lab2 = "Statistical Radius Lower Bound"

    for i in range(N_trials):
        lab = "Monte Carlo distance to moon" if i == 0 else None
        cur_mc_interp = mcs[i][0](times_unique)[:, :3]
        cur_moon_dist = jnp.linalg.norm(cur_mc_interp - jnp.array([1 - mu, 0, 0]), axis=1)
        plt.plot(times_unique*nd.t_star/ 3600 / 24, cur_moon_dist*nd.l_star, alpha=.5, color='purple', label=lab)

    plt.plot(times_unique*nd.t_star/ 3600 / 24, moon_dist*nd.l_star, label="Mean Radius to Moon", linestyle='--')
    plt.plot(t_node_bound*nd.t_star/ 3600 / 24,  col, 'o', label=lab2, color='red', markersize=3)
    plt.plot(times_unique*nd.t_star/ 3600 / 24, d_safe*nd.l_star*jnp.ones_like(times_unique), label="Safe Radius")
    plt.title("Distance to Moon (km)")
    plt.xlabel("Time (days)")
    plt.ylabel("Distance (km)")
    plt.grid()
    plt.legend()

def plot_weights(nodes, data, sf_snopt_sol, nd):
    fig_xi, ax1_xi = plt.subplots()
    xi_time = np.zeros((2*nodes))
    xi_time[::2] = data['times'][0, :]
    xi_time[1::2] = data['times'][-1, :]

    xis_plot = np.zeros((2, 2*nodes))
    xis_plot[:, ::2] = sf_snopt_sol.xStar['xis'].reshape(2, -1)
    xis_plot[:, 1::2] = sf_snopt_sol.xStar['xis'].reshape(2, -1)
    ax1_xi.plot(xi_time*nd.t_star/ 3600 / 24, xis_plot[0, :], label=r"$\xi_r$")
    ax1_xi.plot(xi_time*nd.t_star/ 3600 / 24, xis_plot[1, :], color='red', label=r"$\xi_v$")

    ax1_xi.set_title(r"Weights $\xi$ Over Time")
    ax1_xi.set_xlabel("Time (days)")
    ax1_xi.set_ylabel(r"$\xi$ Value")
    ax1_xi.legend()
    ax1_xi.grid()

#-----------------
# Generate Random Inputs for Testing
#-----------------

def generate_input(y0, yf, nodes):
    us_tmp = jnp.ones(3*nodes)

    y0_tmp = jnp.hstack([y0, 1])
    yf_tmp = jnp.hstack([yf, .95])

    K_ks_tmp = 1e-2*jnp.ones((3*7*nodes))

    input_vals = {'y0': y0_tmp, 'y1': yf_tmp, 'us': us_tmp, 'K_ks': K_ks_tmp}
    return input_vals

def generate_rand_input(y0, yf, nodes):
    us_tmp = jnp.array(np.random.rand(3*nodes))

    y0_tmp = jnp.hstack([y0, 1])
    yf_tmp = jnp.hstack([yf, .95])

    K_ks_tmp = jnp.array(1e-2*np.random.rand((3*7*nodes)))

    input_vals = {'y0': y0_tmp, 'y1': yf_tmp, 'us': us_tmp, 'K_ks': K_ks_tmp}
    return input_vals

#-----------------
# Misc
#-----------------

def process_sparsity(grad_sparsity_orig):
    new_sparsity = {}
    grad_sparsity = grad_sparsity_orig.copy()
    for key, val in grad_sparsity.items():
        cur_obj_constr = val
        new_obj_constr = {}
        for key2, val2_jax in cur_obj_constr.items():
            val2 = np.array(val2_jax)
            if len(val2.shape) != 2:
                if key == 'c_Pyf':
                    new_obj_constr[key2] = val2.reshape(1, -1)
                else:
                    new_obj_constr[key2] = val2.reshape(-1, 1)
            else:
                new_obj_constr[key2] = val2
            
            if jnp.all(new_obj_constr[key2] == 0):
                new_obj_constr.pop(key2, None)
            
        new_sparsity[key] = new_obj_constr
    return new_sparsity

def process_config(config, nd):
    g0 = 9.81 / 1000 # standard gravity
    int_save = config['integration']['int_points']

    # Hot Starter
    read = config['hot_starter']['bool']
    file = config['hot_starter']['file']

    # Tolerances
    R_TOL = config['integration']['r_tol']
    A_TOL = config['integration']['a_tol']

    
    # Segments
    N = config['segments'] # keep odd (probably not needed)
    nodes = N - 1
    forward = np.arange(0, nodes // 2)
    backward = np.flip(np.arange(nodes // 2, nodes))

    # Ephemeris Parameters
    t0 = config['boundary_conditions']['t0']
    tf = config['boundary_conditions']['tf']
    t_node_bound = calc_t_elapsed_nd(t0, tf, N, nd.t_star)


    # Spacecraft Parameters
    m0 = config['engine']['m0']
    Isp = config['engine']['Isp']
    u_max = config['engine']['T_max'] # N

    
    # Uncertainty
    r_1sig = config['uncertainty']['covariance']['initial']['pos_sig'] # km
    v_1sig = config['uncertainty']['covariance']['initial']['vel_sig'] # km/s
    m_1sig = config['uncertainty']['covariance']['initial']['mass_sig'] # kg

    r_1sig_t = config['uncertainty']['covariance']['target']['pos_sig'] # km
    v_1sig_t = config['uncertainty']['covariance']['target']['vel_sig'] # km/s
    m_1sig_t = config['uncertainty']['covariance']['target']['mass_sig'] # kg

    a_err = config['uncertainty']['acc_sig']

    fixed_mag = config['uncertainty']['gates']['fixed_mag'] # fraction
    prop_mag = config['uncertainty']['gates']['prop_mag'] # fraction
    fixed_point = config['uncertainty']['gates']['fixed_point'] # fraction
    prop_point = config['uncertainty']['gates']['prop_point'] # fraction

    eps = config['uncertainty']['eps']

    # Collision Avoidance
    detdet = config['constraints']['deterministic']['det_col_avoid']['bool']
    stochdet = config['constraints']['stochastic']['det_col_avoid']['bool']
    stochstoch = config['constraints']['stochastic']['stat_col_avoid']['bool']

    # Unscented Transform
    alpha_UT = config['UT']['alpha']
    beta_UT = config['UT']['beta']
    kappa_UT = config['UT']['kappa']
    
        # Collision Avoidance
    r_obs = jnp.array(config['constraints']['deterministic']['det_col_avoid']['parameters']['r_obs'])/nd.l_star
    d_safe = jnp.array(config['constraints']['deterministic']['det_col_avoid']['parameters']['safe_d'])/nd.l_star
    
    optOptions = {"Major optimality tolerance": config['SNOPT']['major_opt_tol'],
                  "Major feasibility tolerance": config['SNOPT']['major_feas_tol'],
                  "Minor feasibility tolerance": config['SNOPT']['minor_feas_tol'],
                  'Major iterations limit': config['SNOPT']['major_iter_limit'],
                  'Partial price': config['SNOPT']['partial_price'],
                  'Linesearch tolerance': config['SNOPT']['linesearch_tol'],
                  'Function precision': config['SNOPT']['function_prec'],
                  'Verify level': -1,
                  'Nonderivative linesearch': 1}
    
    T_max_dim = u_max/(nd.a_star*nd.m_star)/1e3

    init_cov = sig2cov(r_1sig, v_1sig, m_1sig, nd)
    targ_cov = sig2cov(r_1sig_t, v_1sig_t, m_1sig_t, nd)

    G_stoch = np.diag(np.array([0, 0, 0, a_err/nd.a_star, a_err/nd.a_star, a_err/nd.a_star, 0])) # stochastic model error

    gates = np.array([fixed_mag, prop_mag, fixed_point, prop_point])

    ve = Isp * g0 / nd.v_star

    if T_max_dim*(t_node_bound[-1] - t_node_bound[0])/(1-1e-2) > ve:
        print("S/C has insufficient mass to continuously thrust:")
        input("Press any key to continue regardless...")
    
    return (int_save, read, file, R_TOL, A_TOL, t0, tf, t_node_bound, N, nodes, forward, backward, m0, Isp, u_max, r_1sig, v_1sig, m_1sig, r_1sig_t, v_1sig_t, m_1sig_t, a_err, fixed_mag, prop_mag, fixed_point, prop_point, eps, detdet, stochdet, stochstoch, alpha_UT, beta_UT, kappa_UT, r_obs, d_safe, optOptions, T_max_dim, init_cov, targ_cov, G_stoch, gates, ve)

def choose_dynamics(config, nd, T_max_dim, ve, d_safe, mu):
    if config['dynamics']['type'] == "2BP":
        dyn_safe = 6378/nd.l_star
        y0 = calc_orbit(config['boundary_conditions']['coe0'], nd)
        yf = calc_orbit(config['boundary_conditions']['coe1'], nd)

        eoms_eval, Aprop_eval, Bprop_eval, Cprop_eval = TwoBodyDynamics(T_max_dim, ve, d_safe)

    elif config['dynamics']['type'] == "CR3BP":
        dyn_safe = 1737.5/nd.l_star
        y0 = jnp.array(config['boundary_conditions']['y0'])
        yf = jnp.array(config['boundary_conditions']['yf'])
        eoms_eval, Aprop_eval, Bprop_eval, Cprop_eval = CR3BPDynamics(T_max_dim, ve, mu, dyn_safe)

    return y0, yf, eoms_eval, Aprop_eval, Bprop_eval, Cprop_eval, dyn_safe

def interpret_bcs(config, t_node_bound, y0, yf, mu, propper):
   
    print("Creating Orbit Interpolants")

    if 'per1' in config['boundary_conditions'].keys():
        tmpt01 = t_node_bound[0]
        tmptf1 = t_node_bound[0] + config['boundary_conditions']['per1']

        tmpt02 = t_node_bound[0] + config['boundary_conditions']['per2']
        tmptf2 = t_node_bound[0]
    else:
        tmpt01 = t_node_bound[0]
        tmptf1 = t_node_bound[-1]

        tmpt02 = t_node_bound[0]
        tmptf2 = t_node_bound[-1]

    alpha_low = config['boundary_conditions']['alpha']['min']
    alpha_high = config['boundary_conditions']['alpha']['max']

    beta_low = config['boundary_conditions']['beta']['min']
    beta_high = config['boundary_conditions']['beta']['max']

    orb0_args = {'t0': tmpt01, 't1': tmptf1, 'a_tol': 1e-12, 'r_tol': 1e-12, 'mu': mu}
    orb0_ys, orb0_ts = propagate_states_perorb(y0, orb0_args, propper)
    orb0_coeff = backward_hermite_coefficients(orb0_ts / jnp.max(orb0_ts), orb0_ys[:, :6])
    y0_interp = CubicInterpolation(orb0_ts / jnp.max(orb0_ts), orb0_coeff)

    orb1_args = {'t1': tmptf2, 't0': tmpt02, 'a_tol': 1e-12, 'r_tol': 1e-12, 'mu': mu}
    orb1_ys, orb1_ts = propagate_states_perorb(yf, orb1_args, propper)
    orb1_coeff = backward_hermite_coefficients(jnp.flip(orb1_ts) / jnp.max(orb1_ts), jnp.flip(orb1_ys[:, :6], axis=0))
    yf_interp = CubicInterpolation(jnp.flip(orb1_ts) / jnp.max(orb1_ts), orb1_coeff)

    return alpha_low, alpha_high, beta_low, beta_high, y0_interp, yf_interp
    
def create_args(config, t_node_bound, y0_inp, yf_inp, R_TOL, A_TOL, gates, init_cov, targ_cov, G_stoch, eps, ve, T_max_dim, nodes, alpha_UT, beta_UT, kappa_UT, r_obs, d_safe):
    args = {'t_node_bound': t_node_bound,
            'y_start': y0_inp,
            'y_end': yf_inp,
            'r_tol': R_TOL, 'a_tol': A_TOL,
            'gates': gates,
            'init_cov': init_cov,
            'targ_cov': targ_cov,
            'inv_targ_cov': np.linalg.inv(targ_cov),
            'inv_targ_cov_sqrt': np.array(mat_sqrt(jnp.linalg.inv(targ_cov))),
            'G_stoch': G_stoch,
            'eps': eps,
            'mx': np.sqrt(chi2.ppf(1 - eps, 3)),
            'nodes': nodes,
            've': ve,
            'T_max_dim': T_max_dim,
            'r_obs': r_obs,
            'd_safe': d_safe,
            'alpha_UT': alpha_UT,
            'beta_UT': beta_UT,
            'kappa_UT': kappa_UT,
            'det_col_avoid_bool': config['constraints']['stochastic']['det_col_avoid']['bool'],
            'stat_col_avoid_bool': config['constraints']['stochastic']['stat_col_avoid']['bool'],
            'int_save': config['integration']['int_points'],
            'free_phasing': True if config['boundary_conditions']['type'] == 'free' else False}
    
    args_state = args_states = {'t_node_bound': t_node_bound,
                   'y_start': y0_inp,
                   'y_end': yf_inp,
            'r_tol': R_TOL, 'a_tol': A_TOL,
            'nodes': nodes,
            've': ve,
            'T_max_dim': T_max_dim,
            'r_obs': r_obs,
            'd_safe': d_safe,
            'det_col_avoid_bool': config['constraints']['deterministic']['det_col_avoid']['bool'],
            'int_save': config['integration']['int_points'],
            'free_phasing': True if config['boundary_conditions']['type'] == 'free' else False}
    
    return args, args_state

def prepare_prop_funcs(propagate_gen, propagate_states_gen, prop_eoms, prop_eoms_states, eoms_eval, Aprop_eval, Bprop_eval, int_save):
    prop_eoms_e = lambda t, c, args: prop_eoms(t, c, args, eoms_eval, Aprop_eval, Bprop_eval)
    prop_eoms_states_e = lambda t, c, args: prop_eoms_states(t, c, args, eoms_eval)

    propagate = lambda y0, args: propagate_gen(y0, args, int_save, prop_eoms_e)
    propagate_states = lambda y0, args: propagate_states_gen(y0, args, int_save, prop_eoms_states_e)


    forward_ode_iteration_e = lambda i, input_dict: forward_ode_iteration(i, input_dict, propagate)
    backward_ode_iteration_e = lambda i, input_dict: backward_ode_iteration(i, input_dict, propagate)

    forward_ode_iteration_states_e = lambda i, input_dict: forward_ode_iteration_states(i, input_dict, propagate_states)
    backward_ode_iteration_states_e = lambda i, input_dict: backward_ode_iteration_states(i, input_dict, propagate_states)

    return prop_eoms_e, prop_eoms_states_e, forward_ode_iteration_e, backward_ode_iteration_e, forward_ode_iteration_states_e, backward_ode_iteration_states_e

def prepare_opt_funcs(args,  args_states, nodes, int_save, forw, back, forw_states, back_states):
    func_states = lambda inps_state: all_constraints_states(inps_state, args_states, nodes, int_save, forw_states, back_states)
    vals_states = jax.jit(jax.block_until_ready(func_states), backend='cpu')
    grad_states = jax.jit(jax.block_until_ready(jax.jacrev(func_states)), backend='cpu')
    sens_states = jax.jit(jax.block_until_ready(lambda inps_state, cvals_state: grad_states(inps_state)), backend='cpu')
    
    func = lambda inps: all_constraints(inps, args, nodes, int_save, forw, back)
    vals = jax.jit(jax.block_until_ready(func), backend='cpu')
    grad = jax.jit(jax.block_until_ready(jax.jacrev(func)), backend='cpu')
    sens = jax.jit(jax.block_until_ready(lambda inps, cvals: grad(inps)), backend='cpu')

    all_constraints_data_e = lambda inps: all_constraints_data(inps, args, nodes, int_save, forw, back)

    return vals, grad, sens, vals_states, grad_states, sens_states, all_constraints_data_e