"""Module to define simulation class."""
import numpy as np


class Simulation:
    """Class for Flow-seq simulation."""

    def __init__(
        self,
        bins,
        diversity,
        size,
        reads,
        fmax,
        distribution,
        ratio_amplification,
        theta1,
        theta2,
        bias_library,
    ):
        """Init.

        Args:
            bins: number of bins in experiment
            diversity: number of different genetic constructs
            size:Number of bacteria sorted
            reads: number of reads in total for your simulation
            fmax: fluorescence max of the FACS
            distribution: fluorescence distribution. either lognormal or gamma
            ratio_amplification: post sorting PCR: What is the PCR amplification ratio?
            theta1: first parameter of the distribution
            (mu for a normal distribution or shape for a gamma distribution)
             in the form of an array
            theta2: second parameter of the distribution
            (sigma for a normal distribution or scale for a gamma distribution)
             in the form of an array
            bias_library: Are some constructs overrepresented in the initial library?
        """
        self.bins = bins
        self.diversity = diversity
        self.size = size
        self.reads = reads  # number of reads in total for your simulation
        self.fmax = fmax
        self.distribution = distribution
        if distribution == "lognormal":
            # Working in log-space
            self.partitioning = np.log(np.logspace(0, np.log10(self.fmax), bins + 1))
        elif distribution == "gamma":
            partitioning = np.logspace(0, np.log10(self.fmax), bins + 1)
            partitioning[0] = 0
            self.partitioning = partitioning
        self.ratio_amplification = (
            ratio_amplification  # post sorting PCR: What is the PCR amplification ratio?
        )
        self.theta1 = theta1  # first parameter of the distribution (mu for a normal distribution or shape for a gamma distribution) in the form of an array
        self.theta2 = theta2  # second parameter of the distribution (sigma for a normal distribution or scale for a gamma distribution) in the form of an array
        self.bias_library = bias_library
