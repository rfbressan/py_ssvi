# Port (attempt) from R code to compute SVI

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from scipy.integrate import quad


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
    @return: A float number representing the rmse
    """
    return np.mean(np.sqrt((raw_svi(param, k)-w)**2))


vol = pd.read_csv("iv.csv").filter(["period", "moneyness", "iv"])  # Select cols
vol = vol[vol["period"] == 30]  # Subset rows where period = 30
vol["tau"] = vol["period"] / 365  # Creates a new column named tau
vol.rename(columns={"moneyness": "k"}, inplace=True)  # rename column to k


# Numerical example as in Jacquier

a, b, rho, m, sigma = 0.030358, 0.0503815, -0.1, 0.3, 0.048922
sviParams = [a, b, rho, m, sigma]
sviParams2 = [a, b, rho, m, 3. * sigma]
xx = np.linspace(-1., 1., 100)

# Testing rmse
w = vol.iv * vol.iv * vol.tau
print(rmse(w, vol.k, sviParams))

impliedVar = np.sqrt(raw_svi(sviParams, xx))
impliedVarpp = np.sqrt(raw_svi(sviParams2, xx))

# Plot IVs
plt.figure(figsize=(7, 3))  # make separate figure
plt.plot(xx, impliedVar, 'b', linewidth=2, label="Standard SVI")
plt.plot(xx, impliedVarpp, 'g', linewidth=2, label="$\sigma$ bumped up")
plt.title("SVI implied volatility smile")
plt.xlabel("log-moneyness", fontsize=12)
plt.legend()
plt.show()

# Plot densities
dens1 = density(sviParams, xx)
dens2 = density(sviParams2, xx)
zero = np.linspace(0.0, 0.0, 100)

plt.figure(figsize=(7, 3))
plt.plot(xx, zero, 'k', linewidth=1)
plt.plot(xx, dens1, 'b', linewidth=2, label="Standard SVI")
plt.plot(xx, dens2, 'g', linewidth=2, label="$\sigma$ bumped up")
plt.title("Implied risk neutral densities")
plt.xlabel("log-moneyness", fontsize=12)
plt.legend()
plt.show()

# Check that density integrates to one
print("Area under denstiy is:",
      quad(lambda x: density(sviParams, x), xx[0], xx[-1])[0])
