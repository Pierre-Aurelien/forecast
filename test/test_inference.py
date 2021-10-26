"""Module to test inference-related functionalitites."""
import numpy as np
import pytest

# mypy: ignore-errors
from forecast.inference import parallel_inference
from forecast.util.experiment import Experiment


@pytest.fixture(name="experiment_data")
def get_class_experiment():
    """Gives experiment class.

    Returns:
        experiment class of sample data
    """
    bins = 16
    cells_bins = np.array(
        [
            787412,
            982020,
            1465090,
            1401737,
            1025836,
            963772,
            1043910,
            1058248,
            1368234,
            1473916,
            2051185,
            2401738,
            2235051,
            1918568,
            1578880,
            412599,
        ]
    ).astype(
        float
    )  # FACS events in each bin ( Number of cells sorted in each bin) Must be a numpy array of dtype=float
    reads = np.array(
        [
            382313.0,
            952717.0,
            701430.0,
            819585.0,
            1074847.0,
            1600514.0,
            2211263.0,
            2471743.0,
            3347620.0,
            3671715.0,
            5220533.0,
            6022885.0,
            5746555.0,
            4967160.0,
            3994495.0,
            1041134.0,
        ]
    ).astype(
        float
    )  # Number of reads allocated in each bin Must be a numpy array of dtype=float
    f_max = 10 ** 6  # Max fluorescence of the FACS
    distribution = "lognormal"  # Fluorescence distribution to choose between lognormal and gamma
    diversity = 244000
    sequencing = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 135, 200, 57, 12, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 7, 1, 26, 92, 243, 13, 27, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 19, 7, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 6, 44, 100, 43, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 76, 112, 2, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 26, 13, 0, 0, 0, 0],
        ]
    )
    return Experiment(bins, diversity, cells_bins, reads, sequencing, f_max, distribution)


def test_size_inference(experiment_data):  # noqa W0621
    """check size of inference.

    Args:
        experiment_data: experiment class

    Returns:
        assess size
    """
    assert parallel_inference(0, 6, experiment_data).shape == (6, 8)
