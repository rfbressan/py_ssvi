# Port (attempt) from R code to compute SVI

import numpy as np
from math import pi
import scipy.optimize as sop


def raw_svi(par, k):
    """
    Returns total variance for a given set of parameters from RAW SVI
    parametrization at given moneyness points.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Total variance
    """
    w = par[0] + par[1] * (par[2] * (k - par[3]) + (
                (k - par[3]) ** 2 + par[4] ** 2) ** 0.5)
    return w


def diff_svi(par, k):
    """
    First derivative of RAW SVI with respect to moneyness.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: First derivative evaluated at k points
    """
    a, b, rho, m, sigma = par
    return b*(rho+(k-m)/(np.sqrt((k-m)**2+sigma**2)))


def diff2_svi(par, k):
    """
    Second derivative of RAW SVI with respect to moneyness.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Second derivative evaluated at k points
    """
    a, b, rho, m, sigma = par
    disc = (k-m)**2 + sigma**2
    return (b*sigma**2)/((disc)**(3/2))


def gfun(par, k):
    """
    Computes the g(k) function. Auxiliary to retrieve implied density and
    essential to test for butterfly arbitrage.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Function g(k) evaluated at k points
    """
    w = raw_svi(par, k)
    w1 = diff_svi(par, k)
    w2 = diff2_svi(par, k)

    g = (1-0.5*(k*w1/w))**2 - (0.25*w1**2)*(w**-1+0.25) + 0.5*w2
    return g


def d1(par, k):
    """
    Auxiliary function to compute d1 from BSM model.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Values of d1 evaluated at k points
    """
    v = np.sqrt(raw_svi(par, k))
    return -k/v + 0.5*v


def d2(par, k):
    """
    Auxiliary function to compute d2 from BSM model.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Values of d2 evaluated at k points
    """
    v = np.sqrt(raw_svi(par, k))
    return -k/v - 0.5*v

def density(par, k):
    """
    Probability density implied by an SVI.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Implied risk neutral probability density from an SVI
    """
    g = gfun(par, k)
    w = raw_svi(par, k)
    dtwo = d2(par, k)

    dens = (g / np.sqrt(2 * pi * w)) * np.exp(-0.5 * dtwo**2)
    return dens


def rmse(w, k, param):
    """
    Returns root mean square error of a RAW SVI parametrization.
    @type w: PdSeries
    @param w: Market total variance
    @param k: Moneyness
    @param param: List of parameters (a, b, rho, m, sigma)
    @return: A float number representing the RMSE
    """
    return np.mean(np.sqrt((raw_svi(param, k)-w)**2))


def wrmse(w, k, param, weights):
    """
    Weighted RMSE
    :param w: Market total variance
    :param k: Moneyness
    :param param: List of parameters (a, b, rho, m, sigma)
    :param weights: Weights. Do not need to sum to 1
    :return: A float number representing the weighted RMSE
    """
    sum_w = np.sum(weights)
    return (1/sum_w) * np.mean(weights*(raw_svi(param, k)-w)**2)


def svi_fit_direct(k, w, weights, method, ngrid=5):
    """
    Direct procedure to calibrate a RAW SVI
    :param k: Moneyness
    :param w: Market total variance
    :param weights: Weights. Do not need to sum to 1
    :param method: Method of optimization. Currently implemented: "brute", "DE"
    :param ngrid: Number of grid points for each parameter in brute force
    algorithm
    :return: SVI parameters (a, b, rho, m, sigma)
    """
    # Bounds on parameters
    bounds = [(0., max(w)),
               (0., 1.),
               (-1., 1.),
               (2*min(k), 2*max(k)),
               (0., 1.)]
    # Tuples for brute force
    bfranges = tuple(bounds)

    # Objective function
    def obj_fun(par):
        sum_w = np.sum(weights)
        return (1 / sum_w) * np.mean(weights * (raw_svi(par, k) - w) ** 2)

    # Chooses algo and run
    if method == "brute":
        # Brute force algo. fmin will take place to refine solution
        p0 = sop.brute(obj_fun, bfranges, Ns=ngrid, full_output=True)
        return p0
    elif method == "DE":
        # Differential Evolution algo.
        p0 = sop.differential_evolution(obj_fun, bounds)
        return p0
    else:
        print("Unknown method passed to svi_fit_direct.")
        raise