"""Main script to visually and quantitatively check coverage of CI."""
import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Check CI coverage.")
    parser.add_argument(
        "--ci_level",
        type=int,
        default=3,
        help="Number of standard deviation to create the confidence interval.",
    )

    parser.add_argument(
        "--metadata_path",
        type=Path,
        help="Folder path containing library data.",
        default="data",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default="out/figure_CI_" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Path for the output folder.",
    )
    parser.add_argument(
        "--fluorescence_amplification",
        type=int,
        default=10,
        help="Ratio fluorescence/protein.",
    )
    args = parser.parse_args()
    return args


def ci_coverage(a, b, c, d):
    """Check if estimate is in 3*confidence interval.

    Args:
        a: estimate
        b: ground truth
        c: confidence interval length
        d: number of confidence interval

    Returns:
        1 (inside) or 0 (outside)
    """
    t = 0
    if a < b - d * c or a > b + d * c:
        t = 0
    else:
        t = 1
    return t


def main():
    """Main script."""
    args = parse_args()

    output_path = args.output_path

    output_path.mkdir(parents=True, exist_ok=True)

    df_estimated = pd.read_csv(Path(args.metadata_path) / "results.csv")
    filter_length = len(df_estimated)
    df_truth = pd.read_csv(Path(args.metadata_path) / "library_gamma.csv")[:filter_length]
    df_truth["B_Protein"] = args.fluorescence_amplification * df_truth["B_Protein"]

    df = pd.concat([df_truth, df_estimated], axis=1)
    df["a_in_CI"] = df.apply(
        lambda row: ci_coverage(row["a_MLE"], row["A_Protein"], row["a_std"], args.ci_level), axis=1
    )
    coverage_a = df["a_in_CI"].mean()
    df["b_in_CI"] = df.apply(
        lambda row: ci_coverage(row["b_MLE"], row["B_Protein"], row["b_std"], args.ci_level), axis=1
    )
    coverage_b = df["b_in_CI"].mean()
    sns.despine()

    fig_dims = (20, 8)
    x = np.linspace(0, 2, filter_length)
    _, ax = plt.subplots(figsize=fig_dims)
    ax.errorbar(
        x,
        (df.sort_values("A_Protein", ascending=False)["a_MLE"]),
        yerr=3 * (df.sort_values("A_Protein", ascending=False)["a_std"]),
        fmt=" ",
        color="tab:brown",
        marker="o",
        markersize=7,
        ecolor="gray",
        elinewidth=2.5,
        capsize=4,
        label="ML estimator",
        zorder=1,
    )
    labl = df.sort_values("A_Protein", ascending=False)["a_in_CI"].to_numpy()
    color = ["tab:orange" if item == 1 else "tab:red" for item in labl]
    ax.scatter(
        x,
        (df.sort_values("A_Protein", ascending=False)["A_Protein"]),
        s=20,
        label="Ground truth",
        color=color,
        zorder=10,
    )
    ax.scatter(
        x,
        (df.sort_values("A_Protein", ascending=False)["a_MOM"]),
        s=50,
        label="MOM estimator",
        c="#4f7942",
        zorder=13,
    )
    ax.legend(frameon=False, fontsize=16, markerscale=1.5)

    ax.axes.get_xaxis().set_visible(False)
    plt.xlabel("Construct Number")
    plt.ylabel("Estimated a parameter", fontsize=16)
    plt.title(f"Confidence Intervals MLE. Coverage is {coverage_a}.")
    plt.savefig(
        Path(output_path) / "a_confidence_interval.png",
        transparent=True,
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()

    fig_dims = (20, 8)
    _, ax = plt.subplots(figsize=fig_dims)
    ax.errorbar(
        x,
        (df.sort_values("B_Protein", ascending=False)["b_MLE"]),
        yerr=2 * (df.sort_values("B_Protein", ascending=False)["a_std"]),
        fmt=" ",
        color="tab:brown",
        marker="o",
        markersize=7,
        ecolor="gray",
        elinewidth=2.5,
        capsize=4,
        label="ML estimator",
        zorder=1,
    )
    labl = df.sort_values("B_Protein", ascending=False)["b_in_CI"].to_numpy()
    color = ["tab:orange" if item == 1 else "tab:red" for item in labl]
    ax.scatter(
        x,
        (df.sort_values("B_Protein", ascending=False)["B_Protein"]),
        s=20,
        label="Ground truth",
        color=color,
        zorder=10,
    )
    ax.scatter(
        x,
        (df.sort_values("B_Protein", ascending=False)["b_MOM"]),
        s=50,
        label="MOM estimator",
        c="#4f7942",
        zorder=13,
    )
    ax.legend(frameon=False, fontsize=16, markerscale=1.5)
    ax.axes.get_xaxis().set_visible(False)
    plt.xlabel("Construct Number")
    plt.ylabel("Estimated b parameter", fontsize=16)
    plt.title(f"Confidence Intervals MLE. Coverage is {coverage_b} ")
    plt.savefig(
        Path(output_path) / "b_confidence_interval.png",
        transparent=True,
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()


if __name__ == "__main__":
    main()
