"""Main script to verify coverage with PCR amplification step."""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.protocol.inference_steps import parallel_inference
from forecast.protocol.simulation_steps import sorting_and_sequencing
from forecast.util.experiment import Experiment
from forecast.util.simulation import Simulation


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Compute CI coverage.")
    parser.add_argument("--rep", type=int, default=1000, help="Number of repetitions.")
    parser.add_argument("--f_max", type=float, default=1e5, help="Fluorescence max of the FACS.")
    parser.add_argument(
        "--distribution", type=str, default="gamma", help="Fluorescence distribution name"
    )
    parser.add_argument(
        "--csv_parameters",
        type=str,
        help="csv file with distribution parameters",
        required=False,
    )
    parser.add_argument("--bins", type=int, default=12, help="Number of bins.")
    parser.add_argument(
        "--size", type=float, default=1e6, help="Number of bacteria sorted trough the FACS."
    )
    parser.add_argument(
        "--reads", type=float, default=1e5, help="Number of reads allocated to sequencing."
    )
    parser.add_argument(
        "--ratio_amplification", type=float, default=1, help="PCR amplification ratio."
    )
    parser.add_argument("--bias_library", type=bool, default=False, help="Bias in the library.")
    parser.add_argument(
        "--metadata_path",
        type=Path,
        help="Folder path containing library data.",
        default="data",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default="out/coverage_data/simulation_" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Path for the output folder.",
    )
    parser.add_argument(
        "--fluorescence_amplification",
        type=int,
        default=10,
        help="Ratio fluorescence/protein.",
    )
    parser.add_argument(
        "--first_index", type=int, default=0, help="Index of first construct to infer."
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=20,
        help="Conduct inference all the way through or just a sample?.",
    )
    args = parser.parse_args()
    return args


def main():  # noqa: CCR001
    """Main script."""
    # parse args
    args = parse_args()

    output_path = args.output_path

    output_path.mkdir(parents=True, exist_ok=True)
    if args.distribution == "gamma":
        name_library = "library_gamma.csv"
        theta1 = "a"
        theta2 = "b"
    elif args.distribution == "lognormal":
        name_library = "library_normal.csv"
        theta1 = "mu"
        theta2 = "sigma"

    df = pd.read_csv(Path(args.metadata_path) / f"{name_library}")

    theta1 = df.iloc[:, 0].to_numpy()
    theta2 = (
        args.fluorescence_amplification * df.iloc[:, 1].to_numpy()
    )  # Fluorescence protein ratio
    diversity = len(theta1)

    # Create an instance of class experiment
    my_simulation = Simulation(
        args.bins,
        diversity,
        args.size,
        args.reads,
        args.f_max,
        args.distribution,
        args.ratio_amplification,
        theta1,
        theta2,
        args.bias_library,
    )
    for rep in range(args.rep):
        sequencing_matrix, sorted_matrix = sorting_and_sequencing(my_simulation)
        np.savetxt(
            args.output_path / f"sequencing_rep_{rep}.csv",
            sequencing_matrix,
            comments="",
            delimiter=",",
        )
        np.savetxt(
            args.output_path / f"cells_bins_rep_{rep}.csv", sorted_matrix[None], delimiter=","
        )

    for rep in range(args.rep):
        cells_bins = (
            pd.read_csv(Path(args.output_path) / f"cells_bins_rep_{rep}.csv", header=None)
            .to_numpy()
            .astype(float)[0]
        )
        sequencing = (
            pd.read_csv(Path(args.output_path) / f"sequencing_rep_{rep}.csv", header=None)
            .to_numpy()
            .astype(int)
        )
        reads = np.sum(sequencing, axis=0)
        diversity = len(sequencing[:, 0])
        bins = int(len(sequencing[0, :]))
        output_path.mkdir(parents=True, exist_ok=True)
        # Experiment Class
        my_experiment = Experiment(
            bins, diversity, cells_bins, reads, sequencing, args.f_max, args.distribution
        )

        parallel_inference(args.first_index, args.end_index, my_experiment).to_csv(
            output_path / f"results_rep_{rep}.csv", index=False
        )


if __name__ == "__main__":
    main()
