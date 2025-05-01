import numpy as np
import jax
import jax.numpy as jnp

from scipy.linalg import ldl

from astropy.time import Time
from astropy import units as u

from scipy.spatial.transform import Rotation as R

#-----------------
# Nondimensionalization
#-----------------

class global_nondim_2B:
    def __init__(self, mu, char_length, char_mass):
        self.mu = mu
        self.l_star = char_length
        self.t_star = np.sqrt(self.l_star ** 3 / self.mu)
        self.v_star = self.l_star/self.t_star
        self.a_star = self.v_star / self.t_star
        self.m_star = char_mass

class global_nondim_CR3BP:
    def __init__(self, Gm1m2, char_length, char_mass):
        self.Gm1m2 = Gm1m2
        self.l_star = char_length
        self.t_star = np.sqrt(self.l_star ** 3 / self.Gm1m2)
        self.v_star = self.l_star/self.t_star
        self.a_star = self.v_star / self.t_star
        self.m_star = char_mass

#-----------------
# Orbital Mechanics
#-----------------

def calc_orbit(orb_elems, nd):
    sma, ecc, inc, lan, argp, ta = orb_elems

    sma_nd = sma / nd.l_star
    p_nd = sma_nd * (1 - ecc**2)

    y0_rv0 = coe2rv(1, p_nd, ecc, inc, lan, argp, ta)

    return jnp.hstack([y0_rv0[0], y0_rv0[1]])

def calc_t_elapsed_nd( t0, tf, N, t_star):
    t0_epoch = Time(t0)
    tf_epoch = Time(tf)

    delta_t = tf_epoch - t0_epoch
    t_elapsed_nd = jnp.linspace(0.0, delta_t.sec, N)/t_star

    return t_elapsed_nd

def calc_t_elapsed_epoch(t0, nd_time, t_star):
    t0_epoch = Time(t0)

    d_times = nd_time*t_star
    t_elapsed_epoch = t0_epoch + (d_times << u.s)

    return t_elapsed_epoch

def prop_eoms_util(t, c, args, eoms_eval):
    states = c[:7]

    state_prop = eoms_eval(t, states, jnp.array([0, 0, 0])).flatten()
    return state_prop

def coe2rv(k, p, ecc, inc, raan, argp, nu):
    # Based on source code from hapsira/poliastro
    pqw = rv_pqw(k, p, ecc, nu)
    rm = coe_rotation_matrix(inc, raan, argp)

    ijk = pqw @ rm.T

    return ijk

def rv_pqw(k, p, ecc, nu):
    # Based on source code from hapsira/poliastro
    pqw = np.array([[np.cos(nu), np.sin(nu), 0], [-np.sin(nu), ecc + np.cos(nu), 0]]) * np.array(
        [[p / (1 + ecc * np.cos(nu))], [np.sqrt(k / p)]]
    )
    return pqw

def coe_rotation_matrix(inc, raan, argp):
    # Based on source code from hapsira/poliastro
    r = rotation_matrix(raan, 2)
    r = r @ rotation_matrix(inc, 0)
    r = r @ rotation_matrix(argp, 2)
    return r

def rotation_matrix(angle, axis):
    # Based on source code from hapsira/poliastro
    assert axis in (0, 1, 2)
    angle = np.asarray(angle)
    c = np.cos(angle)
    s = np.sin(angle)

    a1 = (axis + 1) % 3
    a2 = (axis + 2) % 3
    R = np.zeros(angle.shape + (3, 3))
    R[..., axis, axis] = 1.0
    R[..., a1, a1] = c
    R[..., a1, a2] = -s
    R[..., a2, a1] = s
    R[..., a2, a2] = c
    return R

#-----------------
# JAX functions
#-----------------
def mat_sqrt(mat):
    return jnp.linalg.cholesky(mat) 

def l_max(mat):
    return jnp.linalg.eigvalsh(mat + 1e-12*jnp.diag(jnp.linspace(1.,  2., mat.shape[0])))[-1]

mat_lmax_vec = jax.vmap(l_max, in_axes=2)

def col_avoid(x_cur, r_obs, safe_d):
    dist = x_cur[:3] - r_obs
    return safe_d - jnp.sqrt(dist[0]**2 + dist[1]**2 + dist[2]**2)

col_avoid_vmap = jax.vmap(col_avoid, in_axes=(0, None, None))

def UT_col_avoid(mean, cov, r_obs, safe_d, alpha_UT, beta_UT, kappa_UT):

    # Unscented Transform

    # Start
    nx = cov.shape[0]

    lambda_UT = alpha_UT ** 2 * (nx + kappa_UT) - nx

    # Pick Sample Points
    sigmas = jnp.zeros((2 * nx + 1, nx))
    weights_m = jnp.zeros(2 * nx + 1)
    weights_c = jnp.zeros(2 * nx + 1)

    sigmas = sigmas.at[:].set(mean)
    weights_m = weights_m.at[0].set(lambda_UT / (nx + lambda_UT))
    weights_c = weights_c.at[0].set(lambda_UT / (nx + lambda_UT) + 1 - alpha_UT ** 2 + beta_UT)

    weights_m = weights_m.at[1:].set(1 / (2 * (nx + lambda_UT)))
    weights_c = weights_c.at[1:].set(1 / (2 * (nx + lambda_UT)))

    term_sq =  (nx + lambda_UT) * cov
    # term_sp =  sp_spr.csr_matrix(term_sq)
    term = mat_sqrt(term_sq)

    sigmas = sigmas.at[1:1+nx, :].add(term[:, 0:nx])
    sigmas = sigmas.at[nx + 1:2*nx + 1, :].add(-term[:, 0:nx])

    # for i in range(nx):
    #
    #     sigmas[i + 1] = mean + term[:, i]
    #     sigmas[nx + i + 1] = mean - term[:, i]

    col_avoid_vals = col_avoid_vmap(sigmas, r_obs, safe_d)

    ybar = jnp.sum(weights_m*col_avoid_vals)
    Py = jnp.sum(weights_c*(col_avoid_vals - ybar) ** 2)

    return ybar, Py, weights_m, weights_c, sigmas

UT_col_avoid_vmap = jax.vmap(UT_col_avoid, in_axes=(0, 2, None, None, None, None, None))

def ABAT(A, B):
    return A @ B @ A.T

vec_ABAT = jax.vmap(ABAT, in_axes=(2, 2), out_axes=2)

def init_stat_zeros(nodes, int_save):
    states = jnp.zeros((int_save, 7+7**2+7*3+7**2, nodes))
    times = jnp.zeros((int_save, nodes))

    A_ks = jnp.zeros((7, 7, nodes))
    B_ks = jnp.zeros((7, 3, nodes))

    K_ks = jnp.zeros((3, 6, nodes))

    sig_k = jnp.zeros((7, 7, nodes))

    P_ks = jnp.zeros((7, 7, nodes+1))
    return states, times, A_ks, B_ks, K_ks, sig_k, P_ks

#-----------------
# Numpy functions
#-----------------

def LDL_sqrt(mat):
    mat = np.array(mat)
    mat[mat < 0] = 1e-12
    L, D, _ = ldl(mat)
    return L@np.sqrt(mat)

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def A2q(A):
    mat = R.from_matrix(A)
    return mat.as_quat()

def sig2cov(r_1sig, v_1sig, m_1sig, nd):
    r_cov = (r_1sig/nd.l_star)**2
    v_cov = (v_1sig/nd.v_star)**2
    m_cov = (m_1sig/nd.m_star)**2

    return np.diag(np.array([r_cov, r_cov, r_cov, v_cov, v_cov, v_cov, m_cov]))