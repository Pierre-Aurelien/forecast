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
        "--distribution", type=str, default="gamma", help="Fluorescence distribution name"
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
        default=1,
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

    if args.distribution == "gamma":
        name_library = "library_gamma.csv"
        theta1 = "a"
        theta2 = "b"
    elif args.distribution == "lognormal":
        name_library = "library_normal.csv"
        theta1 = "mu"
        theta2 = "sigma"

    df_truth = pd.read_csv(Path(args.metadata_path) / f"{name_library}")
    df_truth[f"{theta2}"] = args.fluorescence_amplification * df_truth[f"{theta2}"]

    df = pd.concat([df_truth, df_estimated], axis=1)

    # quality filtering
    df = df[df["Score"] <= 0.7]  # Drop constructs too much on the border
    df = df[df["Inference_grade"] == 1]  # Keep constructs with good shape
    df = df[df[f"{theta1}_std"] < df[f"{theta1}_MLE"]]
    df = df[df[f"{theta2}_std"] < df[f"{theta2}_MLE"]]

    df[f"{theta1}_in_CI"] = df.apply(
        lambda row: ci_coverage(
            row[f"{theta1}_MLE"], row[f"{theta1}"], row[f"{theta1}_std"], args.ci_level
        ),
        axis=1,
    )
    coverage_a = df[f"{theta1}_in_CI"].mean()
    df[f"{theta2}_in_CI"] = df.apply(
        lambda row: ci_coverage(
            row[f"{theta2}_MLE"], row[f"{theta2}"], row[f"{theta2}_std"], args.ci_level
        ),
        axis=1,
    )
    coverage_b = df[f"{theta2}_in_CI"].mean()
    sns.despine()

    fig_dims = (20, 8)
    x = np.linspace(0, 2, len(df))
    _, ax = plt.subplots(figsize=fig_dims)
    ax.errorbar(
        x,
        (df.sort_values(f"{theta1}", ascending=False)[f"{theta1}_MLE"]),
        yerr=3 * (df.sort_values(f"{theta1}", ascending=False)[f"{theta1}_std"]),
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
    labl = df.sort_values(f"{theta1}", ascending=False)[f"{theta1}_in_CI"].to_numpy()
    color = ["tab:orange" if item == 1 else "tab:red" for item in labl]
    ax.scatter(
        x,
        (df.sort_values(f"{theta1}", ascending=False)[f"{theta1}"]),
        s=20,
        label="Ground truth",
        color=color,
        zorder=10,
    )
    ax.scatter(
        x,
        (df.sort_values(f"{theta1}", ascending=False)[f"{theta1}_MOM"]),
        s=50,
        label="MOM estimator",
        c="#4f7942",
        zorder=13,
    )
    ax.legend(frameon=False, fontsize=16, markerscale=1.5)

    ax.axes.get_xaxis().set_visible(False)
    plt.xlabel("Construct Number")
    plt.ylabel(f"Estimated {theta1}  parameter", fontsize=16)
    plt.title(f"Confidence Intervals MLE. Coverage is {coverage_a}.")
    plt.savefig(
        Path(output_path) / f"{theta1}_confidence_interval.png",
        transparent=True,
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()

    fig_dims = (20, 8)
    _, ax = plt.subplots(figsize=fig_dims)
    ax.errorbar(
        x,
        (df.sort_values(f"{theta2}", ascending=False)[f"{theta2}_MLE"]),
        yerr=2 * (df.sort_values(f"{theta2}", ascending=False)[f"{theta2}_std"]),
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
    labl = df.sort_values(f"{theta2}", ascending=False)[f"{theta2}_in_CI"].to_numpy()
    color = ["tab:orange" if item == 1 else "tab:red" for item in labl]
    ax.scatter(
        x,
        (df.sort_values(f"{theta2}", ascending=False)[f"{theta2}"]),
        s=20,
        label="Ground truth",
        color=color,
        zorder=10,
    )
    ax.scatter(
        x,
        (df.sort_values(f"{theta2}", ascending=False)[f"{theta2}_MOM"]),
        s=50,
        label="MOM estimator",
        c="#4f7942",
        zorder=13,
    )
    ax.legend(frameon=False, fontsize=16, markerscale=1.5)
    ax.axes.get_xaxis().set_visible(False)
    plt.xlabel("Construct Number")
    plt.ylabel(f"Estimated {theta2} parameter", fontsize=16)
    plt.title(f"Confidence Intervals MLE. Coverage is {coverage_b} ")
    plt.savefig(
        Path(output_path) / f"{theta2}_confidence_interval.png",
        transparent=True,
        bbox_inches="tight",
        dpi=600,
    )
    plt.close()


if __name__ == "__main__":
    main()
