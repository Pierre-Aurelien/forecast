"""Module to define Experiment class."""
import numpy as np


class Experiment:
    """Class for Flow-seq experiment."""

    def __init__(self, bins, diversity, nj, reads, sequencing, fmax, distribution):
        """Init.

        Args:
            bins: number of bins
            diversity: number of different genetic constructs
            nj:FACS events in each bin ( Number of cells sorted in each bin)
            reads:#Number of reads allocated in each bin
            sequencing: Filtered Read Counts for each genetic construct
            (one row) in each bin (one column)
            fmax:fluorescence max of the FACS
            distribution:fluorescence distribution. either lognormal or gamma
        """
        self.bins = bins
        self.diversity = diversity
        self.nj = nj
        self.size = np.sum(self.nj)
        self.reads = reads
        self.sequencing = sequencing
        self.fmax = fmax
        self.distribution = distribution
        if distribution == "lognormal":
            # Working in log-space
            self.partitioning = np.log(np.logspace(0, np.log10(fmax), bins + 1))
        elif distribution == "gamma":
            # Working in normal fluorescence space
            partitioning = np.logspace(0, np.log10(fmax), bins + 1)
            partitioning[0] = 0
            self.partitioning = partitioning
        self.mean_assigned = [
            (self.partitioning[j + 1] + self.partitioning[j]) / 2 for j in range(bins)
        ]
        self.enrich = np.divide(nj, reads, out=np.zeros_like(nj), where=reads != 0, dtype=float)
        self.nijhat = np.multiply(self.sequencing, self.enrich).astype(int)
        self.nihat = self.nijhat.sum(axis=1)
