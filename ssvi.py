# Port (attempt) from R code to compute SSVI

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import animation
#from matplotlib.colors import cnames
from math import pi
#from scipy.integrate import quad


def phi_fun(par, theta, phitype):
    """
    Phi function, auxiliary to compute a SSVI. Type can be either "heston" (
    default) or "powerlaw", in which case par must have two values, one for
    gamma and the second for eta.
    @param par: Complete list of parameters for the SSVI function in that
    order, (rho, sigma, gamma, [eta]). For "heston" type only gamma will be
    selected, for "powerlaw", gamma and eta
    @param theta: ATM implied total variance
    @param phitype: Text string representing the type of phi function. One of
    "heston" or "powerlaw".
    @return: Values for the chosen type of phi function
    """
    if phitype == "powerlaw":
        gamma, eta = par[-2:]
        return eta*theta**-gamma
    else:
        gamma = par[-2]
        return 1. / (gamma*theta)*(1.-(1.-np.exp(-gamma*theta))/(gamma*theta))


def ssvi_fun(par, t, k, phitype):
    """
    SSVI function
    @param par: Parameters of the function in that order, (rho, sigma, gamma,
    [eta])
    @param t: Time to maturity
    @param k: Forward log-moneyness
    @param phitype: Text string representing the type of phi function. One of
    "heston" or "powerlaw".
    @return: Values of SSVI function for the pair (theta, k), considering
    that $\theta := \sigma^2 t$.
    """
    theta = par[1]**2 * t

    p = phi_fun(par, theta, phitype)

    return 0.5*theta*(1.+par[0]*p*k+np.sqrt((p*k+par[0])**2+1-par[0]**2))


def ssvi_diff(par, t, k, phitype):
    """
    First derivative of SSVI with respect to k
    @param par: Parameters of the function in that order, (rho, sigma, gamma,
    [eta])
    @param t: Time to maturity
    @param k: Forward log-moneyness
    @param phitype: Text string representing the type of phi function. One of
    "heston" or "powerlaw".
    @return: Values of diff(SSVI, k) for the pair (theta, k), considering
    that $\theta := \sigma^2 t$.
    """
    theta = par[1] ** 2 * t
    p = phi_fun(par, theta, phitype)
    rho = par[0]
    pkr = p*k+rho

    return 0.5*theta*p*(rho + pkr/np.sqrt(pkr**2+1-rho**2))


def ssvi_diff2(par, t, k, phitype):
    """
    Second derivative of SSVI with respect to k
    @param par: Parameters of the function in that order, (rho, sigma, gamma,
    [eta])
    @param t: Time to maturity
    @param k: Forward log-moneyness
    @param phitype: Text string representing the type of phi function. One of
    "heston" or "powerlaw".
    @return: Values of diff(SSVI, k, k) for the pair (theta, k), considering
    that $\theta := \sigma^2 t$.
    """
    theta = par[1] ** 2 * t
    p = phi_fun(par, theta, phitype)
    rho = par[0]
    pkr = p * k + rho

    return 0.5*p**2*theta*(1 - rho**2)/(pkr**2 + 1 - rho**2)**(3/2)

def ssvi_difft(par, t, k, phitype):
    """
    First derivative of SSVI with respect to t
    @param par: Parameters of the function in that order, (rho, sigma, gamma,
    [eta])
    @param t: Time to maturity
    @param k: Forward log-moneyness
    @param phitype: Text string representing the type of phi function. One of
    "heston" or "powerlaw".
    @return: Values of diff(SSVI, t) for the pair (theta, k), considering
    that $\theta := \sigma^2 t$.
    """
    # Finite difference method
    eps = 1e-6
    return (ssvi_fun(par, t+eps, k, phitype) - ssvi_fun(par, t-eps, k,
                                                        phitype)) / (2*eps)

def ssvi_g(par, t, k, phitype):
    """
    Computes the g(k) function from an SSVI parameters
    @param par: Parameters of the function in that order, (rho, sigma, gamma,
    [eta])
    @param t: Time to maturity
    @param k: Forward log-moneyness
    @param phitype: Text string representing the type of phi function. One of
    "heston" or "powerlaw".
    @return: The g(theta, k) function, considering that $\theta := \sigma^2 t$.
    """
    w = ssvi_fun(par, t, k, phitype)
    w1 = ssvi_diff(par, t, k, phitype)
    w2 = ssvi_diff2(par, t, k, phitype)

    return (1-0.5*(k*w1/w))**2 - (0.25*w1**2)*(w**-1+0.25) + 0.5*w2


def ssvi_d1(par, t, k, phitype):
    """
    Auxiliary function to compute d1 from BSM model.
    @param par: Parameters of the function in that order, (rho, sigma, gamma,
    [eta])
    @param t: Time to maturity
    @param k: Forward log-moneyness
    @param phitype: Text string representing the type of phi function. One of
    "heston" or "powerlaw".
    @return: Returns d1 from BSM model evaluated at (k, w) points
    """
    v = np.sqrt(ssvi_fun(par, t, k, phitype))
    return -k/v + 0.5 * v


def ssvi_d2(par, t, k, phitype):
    """
    Auxiliary function to compute d2 from BSM model.
    @param par: Parameters of the function in that order, (rho, sigma, gamma,
    [eta])
    @param t: Time to maturity
    @param k: Forward log-moneyness
    @param phitype: Text string representing the type of phi function. One of
    "heston" or "powerlaw".
    @return: Returns d2 from BSM model evaluated at (k, w) points
    """
    v = np.sqrt(ssvi_fun(par, t, k, phitype))
    return -k/v - 0.5 * v


def ssvi_density(par, t, k, phitype):
    """
    Probability density implied by an SSVI.
    @param par: Parameters of the function in that order, (rho, sigma, gamma,
    [eta])
    @param t: Time to maturity
    @param k: Forward log-moneyness
    @param phitype: Text string representing the type of phi function. One of
    "heston" or "powerlaw".
    @return: Implied risk neutral probability density from a SSVI
    """
    g = ssvi_g(par, t, k, phitype)
    w = ssvi_fun(par, t, k, phitype)
    dtwo = ssvi_d2(par, t, k, phitype)

    dens = (g / np.sqrt(2 * pi * w)) * np.exp(-0.5 * dtwo**2)
    return dens


def ssvi_local_vol(par, t, k, phitype):
    """
    Compute Local Volatility function through Dupire's equation for a SSVI
    parametrization.
    @param par: Parameters of the function in that order, (rho, sigma, gamma,
    [eta])
    @param t: Time to maturity
    @param k: Forward log-moneyness
    @param phitype: Text string representing the type of phi function. One of
    "heston" or "powerlaw".
    @return: Local volatility function evaluated at (k, t) points
    """

    return np.sqrt(ssvi_difft(par, t, k, phitype) / ssvi_g(par, t, k, phitype))


# Numerical examples as in Jacquier
rho, sigma, gamma, eta = -0.7, 0.2, 0.8, 1.
par_heston = [rho, sigma, gamma]
par_pl = [rho, sigma, gamma, eta]
t = 0.1
xx, TT = np.linspace(-1., 1., 50), np.linspace(0.001, 5., 50)

print("Arbitrage avoided in heston: ", gamma - 0.25*(1.+np.abs(rho)) >= 0.0)

loc_vol = [[ssvi_local_vol(par_heston, t, k, "heston") for k in xx] for t in TT]
loc_vol_np = np.array(loc_vol)

# Plot Local Volatility Surface
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection="3d")
xxx, TTT = np.meshgrid(xx, TT)
ax.plot_surface(xxx, TTT, loc_vol_np, cmap=plt.cm.jet, rstride=1, cstride=1,
                linewidth=0)
ax.set_xlabel("Forward log-moneyness")
ax.set_ylabel("Time to maturity")
ax.set_zlabel("Local volatility")
ax.set_title("SSVI Local Volatility")
plt.show()

