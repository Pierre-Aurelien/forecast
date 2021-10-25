"""Module to define useful transformations."""
import numpy as np
import scipy.stats as stats
from scipy import integrate, special


def log_moments_mu(m, s):
    """Computes the mean of a log_normal distribution.

    Args:
        m: first parameter of log-normal distribution
        s: second parameter of log-normal distribution

    Returns:
        mean of the log_normal distribution
    """
    return np.exp(m + 0.5 * (s ** 2))


def log_moments_sigma(m, s):
    """Computes the variance of a log_normal distribution.

    Args:
        m: first parameter of log-normal distribution
        s: second parameter of log-normal distribution

    Returns:
      variance of the log_normal distribution
    """
    # Takes as input the parameters m and s of the log normal distribution and outputs the variance
    return np.exp(2 * m + (s ** 2)) * (np.exp(s ** 2) - 1)


def get_mape(x, x_hat):
    """Computes the Mean Absolute Percentage Error.

    Args:
        x: ground truth
        x_hat: estimates

    Returns:
        MAPE

    """
    return np.abs(x - x_hat) / x


def get_wd(m, s, m0, s0):
    """Computes 1-Wasserstein distance between the two normals.

    Args:
        m: mean of first normal distribution
        s: variance of first normal distribution
        m0: mean of second normal distribution
        s0: variance of second normal distribution

    Returns:
        1-Wasserstein distance between the two normals

    """

    def f(x):
        np.abs(m - m0 + np.sqrt(2) * (s - s0) * special.erfinv(x))

    return integrate.quad(f, 0, 1)


def visibility(m, s, fm):
    """Computes the probability mass of a normal distribution in a upper bounded domain.

    Args:
        m: mean of first normal distribution
        s: variance of first normal distribution
        fm: fluorescence upper bound

    Returns:
        Computes the probability mass up to fm of the normal distribution parameterised by m and s
    """
    return stats.norm.cdf(fm, loc=m, scale=s) - stats.norm.cdf(0, loc=m, scale=s)


def ab_to_ms(a, b):
    """Convert shape and scale of gamma distribution to mean and standard deviation.

    Args:
        a: shape of gamma distribution
        b: scale of gamma distribution

    Returns:
        mean and standard deviation
    """
    return np.array([a * b, np.sqrt(a) * b])


def ms_to_ab(m, s) -> np.ndarray:
    """Convert mean and standard deviation of gamma distribution to shape and scale.

    Args:
        m: mean
        s: standard deviation

    Returns:
        shape and scale parameters
    """
    # takes as input the mean and standard deviation of the gamma dsitribution and return shape and scale parameters
    return np.array([(m / s) ** 2, s ** 2 / m])
