"""Main script to execute inference."""
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from forecast.inference import parallel_inference
from forecast.util.experiment import Experiment


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Inference of Flow-Seq dataset.")
    parser.add_argument("--f_max", type=float, default=1e5, help="Fluorescence max of the FACS.")
    parser.add_argument(
        "--distribution", type=str, default="lognormal", help="Fluorescence distribution name"
    )
    parser.add_argument(
        "--csv_sequencing",
        type=str,
        help="csv file from sequencing",
        required=False,
    )
    parser.add_argument(
        "--csv_cells_bins",
        type=str,
        help="csv file giving the number of cells per bin",
        required=False,
    )
    parser.add_argument(
        "--csv_reads",
        type=str,
        help="csv file of number of reads per bin",
        required=False,
    )
    parser.add_argument(
        "--metadata_path",
        type=Path,
        help="Folder path containing all results file.",
        default="data",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default="out/inference_results" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Path for the output folder.",
    )
    args = parser.parse_args()
    return args


def main():  # noqa: CCR001
    """Main script."""
    # parse args
    args = parse_args()

    output_path = args.output_path
    cells_bins = (
        pd.read_csv(Path(args.metadata_path) / "cells_bins.csv", header=None)
        .to_numpy()
        .astype(float)[0]
    )
    print("cells bins are", cells_bins)
    reads = (
        pd.read_csv(Path(args.metadata_path) / "reads.csv", header=None).to_numpy().astype(float)[0]
    )
    sequencing = (
        pd.read_csv(Path(args.metadata_path) / "sequencing.csv", header=None).to_numpy().astype(int)
    )

    diversity = len(sequencing[:, 0])
    bins = int(len(sequencing[0, :]))
    output_path.mkdir(parents=True, exist_ok=True)
    # Experiment Class
    my_experiment = Experiment(
        bins, diversity, cells_bins, reads, sequencing, args.f_max, args.distribution
    )
    print(my_experiment.nj)
    print("ive arrived here")

    parallel_inference(0, 10, my_experiment)


if __name__ == "__main__":
    main()
