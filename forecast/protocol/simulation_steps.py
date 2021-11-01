"""Module for conducting inference on MPRA data."""
import random
from typing import Tuple

import numpy as np

from forecast.util.simulation import Simulation


def sorting_and_sequencing(simulation: Simulation) -> Tuple[np.ndarray, np.ndarray]:
    """Perform sorting and sequencing steps.

    take as input the number of bins (simulation.bins), the diversity (simulation.diversity),
    size of the library sorted (n), the number of reads to allocate in total (simulation.reads),
    the post sorting amplification step (simulation.ratio_amplification),
    if the library is balanced (BIAS_Library),
    the underlying protein simulation.distribution (gamma or lognormal),
    the fluorescence bounds for the sorting machine (simulation.partitioning),
    and the parameters of the said simulation.distribution.
    Args:
        simulation: instance of the simulation class

    Returns:
        Return the (simulation.diversity*Bins) matrix resulting from the sequencing
        and the sorting matrix n (number of cell sorted in each bin)
    """
    #### STEP 1 - Draw the ratio p_concentration
    if simulation.bias_library:
        params = np.ones(simulation.diversity)
        dirichlet_sample = [random.gammavariate(a, 1) for a in params]
        p_concentration = [v / sum(dirichlet_sample) for v in dirichlet_sample]
        # Sample from the simulation.diversity simplex to get ratios
        # p_concentration=np.ones(simulation.diversity)/simulation.diversity
    else:
        p_concentration = [1 / simulation.diversity] * simulation.diversity

    #### STEP 2 - Draw the sample sizes= of each genetic construct
    ni = np.random.multinomial(simulation.size, p_concentration, size=1)[0]

    #### STEP 3 - Compute binning
    nij = np.empty((simulation.diversity, simulation.bins))
    for i in range(simulation.diversity):
        e = np.random.gamma(simulation.theta1[i], simulation.theta2[i], ni[i])
        nij[i, :] = np.histogram(e, bins=simulation.partitioning)[0]

    nij = nij.astype(int)

    #### STEP 4 - PCR amplification
    nij_amplified = np.multiply(nij, simulation.ratio_amplification)

    #### STEP 5 - Compute Reads allocation
    n = np.sum(nij)
    nj = np.sum(nij, axis=0)
    reads = np.floor(
        nj * simulation.reads / n
    )  # Allocate reads with respect to the number of cells sorted in each bin

    #### STEP 6 - DNA sampling
    sij = np.zeros((simulation.diversity, simulation.bins))

    # Compute ratios& Multinomial sampling
    for j in range(simulation.bins):
        if np.sum(nij_amplified, axis=0)[j] != 0:
            concentration_vector = nij_amplified[:, j] / np.sum(nij_amplified, axis=0)[j]
        else:
            concentration_vector = np.zeros(simulation.diversity)
        sij[:, j] = np.random.multinomial(reads[j], concentration_vector, size=1)
    return (sij, nj)


def sorting(simulation: Simulation) -> np.ndarray:
    """Perform the sorting step.

    Args:
        simulation: instance of the simulation class

    Returns:
        the (simulation.diversity*Bins) matrix resulting from the sorting step

    """
    #### STEP 1 - Draw the ratio p_concentration
    if simulation.bias_library:
        params = np.ones(simulation.diversity)
        dirichlet_sample = [random.gammavariate(a, 1) for a in params]
        p_concentration = [v / sum(dirichlet_sample) for v in dirichlet_sample]
    else:
        p_concentration = [1 / simulation.diversity] * simulation.diversity

    #### STEP 2 - Draw the sample sizes= of each genetic construct
    ni = np.random.multinomial(simulation.size, p_concentration, size=1)[0]

    #### STEP 3 - Compute binning
    nij = np.empty((simulation.diversity, simulation.bins))
    for i in range(simulation.diversity):
        e = np.random.gamma(simulation.theta1[i], simulation.theta2[i], ni[i])
        nij[i, :] = np.histogram(e, bins=simulation.partitioning)[0]

    nij = nij.astype(int)
    return nij


def sequencing(simulation: Simulation, nij: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform sequencing step.

    Args:
        simulation: instance of the Simulation class
        nij: Matrix resulting from the sorting step

    Returns:
        Return the (simulation.diversity*Bins) matrix resulting from the sequencing sij
        and the sorting matrix n (number of cell sorted in each bin)

    """
    #### STEP 4 - PCR amplification
    nij_amplified = np.multiply(nij, simulation.ratio_amplification)

    #### STEP 5 - Compute Reads allocation
    n = np.sum(nij)
    nj = np.sum(nij, axis=0)
    reads = np.floor(
        nj * simulation.reads / n
    )  # Allocate reads with respect to the number of cells srted in each bin
    #### STEP 6 - DnA sampling

    sij = np.zeros((simulation.diversity, simulation.bins))

    # Compute ratios& Multinomial sampling
    for j in range(simulation.bins):
        if np.sum(nij_amplified, axis=0)[j] != 0:
            concentration_vector = nij_amplified[:, j] / np.sum(nij_amplified, axis=0)[j]
        else:
            concentration_vector = np.zeros(simulation.diversity)
        sij[:, j] = np.random.multinomial(reads[j], concentration_vector, size=1)
    return sij, nj
