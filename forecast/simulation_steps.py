"""Module for conducting inference on MPRA data."""
import random
from typing import Tuple

import numpy as np
import scipy.stats as stats

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

    def sorting_protein_matrix_populate(i, j):
        if simulation.distribution == "lognormal":
            element_matrix = stats.norm.cdf(
                simulation.partitioning[j + 1], loc=simulation.theta1[i], scale=simulation.theta2[i]
            ) - stats.norm.cdf(
                simulation.partitioning[j], loc=simulation.theta1[i], scale=simulation.theta2[i]
            )
        else:
            element_matrix = stats.gamma.cdf(
                simulation.partitioning[j + 1], a=simulation.theta1[i], scale=simulation.theta2[i]
            ) - stats.gamma.cdf(
                simulation.partitioning[j], a=simulation.theta1[i], scale=simulation.theta2[i]
            )
        return element_matrix

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

    ## Compute ratios qji
    qij = np.fromfunction(
        sorting_protein_matrix_populate, (simulation.diversity, simulation.bins), dtype=int
    )

    ## Compute nij
    nij = qij * ni[:, np.newaxis]
    nij = np.floor(nij)  # Convert to Integer numbers

    #### STEP 4 - PCR amplification

    nij_amplified = np.multiply(nij, simulation.ratio_amplification)

    #### STEP 5 - Compute Reads allocation
    n = np.sum(nij)
    n = np.sum(nij, axis=0)
    reads = np.floor(
        n * simulation.reads / n
    )  # Allocate reads with repsect to the number of cells srted in each bin
    #### STEP 6 - DnA sampling

    sij = np.zeros((simulation.diversity, simulation.bins))

    # Compute ratios& Multinomial sampling
    for j in range(simulation.bins):
        if np.sum(nij_amplified, axis=0)[j] != 0:
            concentration_vector = nij_amplified[:, j] / np.sum(nij_amplified, axis=0)[j]
        else:
            concentration_vector = np.zeros(simulation.diversity)
        sij[:, j] = np.random.multinomial(reads[j], concentration_vector, size=1)
    return (sij, n)


def sorting(simulation: Simulation) -> np.ndarray:
    """Perform the sorting step.

    Args:
        simulation: instance of the simulation class

    Returns:
        the (simulation.diversity*Bins) matrix resulting from the sorting step

    """
    #### STEP 1 - Draw the ratio p_concentration

    def sorting_protein_matrix_populate(i, j):
        if simulation.distribution == "lognormal":
            element_matrix = stats.norm.cdf(
                simulation.partitioning[j + 1], loc=simulation.theta1[i], scale=simulation.theta2[i]
            ) - stats.norm.cdf(
                simulation.partitioning[j], loc=simulation.theta1[i], scale=simulation.theta2[i]
            )
        else:
            element_matrix = stats.gamma.cdf(
                simulation.partitioning[j + 1], a=simulation.theta1[i], scale=simulation.theta2[i]
            ) - stats.gamma.cdf(
                simulation.partitioning[j], a=simulation.theta1[i], scale=simulation.theta2[i]
            )
        return element_matrix

    if simulation.bias_library:
        params = np.ones(simulation.diversity)
        dirichlet_sample = [random.gammavariate(a, 1) for a in params]
        p_concentration = [v / sum(dirichlet_sample) for v in dirichlet_sample]
    else:
        p_concentration = [1 / simulation.diversity] * simulation.diversity

    #### STEP 2 - Draw the sample sizes= of each genetic construct

    ni = np.random.multinomial(simulation.size, p_concentration, size=1)[0]
    # ni=ni[0]

    #### STEP 3 - Compute binning

    ## Compute ratios qji
    qij = np.fromfunction(
        sorting_protein_matrix_populate, (simulation.diversity, simulation.bins), dtype=int
    )

    ## Compute nij
    nij = qij * ni[:, np.newaxis]
    nij = np.floor(nij)  # Convert to Integer numbers
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
    n = np.sum(nij, axis=0)
    reads = np.floor(
        n * simulation.reads / n
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
    return sij, n
