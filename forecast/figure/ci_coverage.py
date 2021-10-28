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
    parser.add_argument("--f_max", type=float, default=1e5, help="Fluorescence max of the FACS.")
    parser.add_argument(
        "--distribution", type=str, default="gamma", help="Fluorescence distribution name"
    )
    parser.add_argument(
        "--csv_ground_truth",
        type=str,
        help="csv file with ground truth distribution parameters",
        required=False,
    )
    parser.add_argument(
        "--csv_estimated",
        type=str,
        help="csv file with estimated distribution parameters",
        required=False,
    )
    parser.add_argument("--bins", type=int, default=12, help="Number of bins.")
    parser.add_argument(
        "--size", type=float, default=1e6, help="Number of bacteria sorted trough the FACS."
    )
    parser.add_argument(
        "--reads", type=float, default=1e7, help="Number of reads allocated to sequencing."
    )
    parser.add_argument(
        "--ratio_amplification", type=float, default=1e2, help="PCR amplification ratio."
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
        default="out/figure_CI_" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Path for the output folder.",
    )
    parser.add_argument(
        "--fluorescence_amplification",
        type=int,
        default=100,
        help="Ratio fluorescence/protein.",
    )
    args = parser.parse_args()
    return args


def ci_coverage(a, b, c):
    """Check if estimate is in 2*confidence interval.

    Args:
        a: estimate
        b: ground truth
        c: confidence interval length

    Returns:
        1 (inside) or 0 (outside)
    """
    t = 0
    if a < b - 2 * c or a > b + 2 * c:
        t = 0
    else:
        t = 1
    return t


def main():
    """Main script."""
    args = parse_args()

    output_path = args.output_path

    output_path.mkdir(parents=True, exist_ok=True)

    df_truth = pd.read_csv(Path(args.metadata_path) / "library_gamma.csv")
    df_estimated = pd.read_csv(Path(args.metadata_path) / "results.csv")

    df = pd.concat([df_truth, df_estimated], axis=1)
    df["a_in_CI"] = df.apply(
        lambda row: ci_coverage(row["a_MLE"], row["A_Protein"], row["a_std"]), axis=1
    )
    df["b_in_CI"] = df.apply(
        lambda row: ci_coverage(row["b_MLE"], row["B_Protein"], row["b_std"]), axis=1
    )

    sns.despine()

    fig_dims = (20, 8)
    x = np.linspace(0, 2, 50)
    _, ax = plt.subplots(figsize=fig_dims)
    ax.errorbar(
        x,
        (df.sort_values("A_Protein", ascending=False)["a_MLE"])[160:210],
        yerr=2 * (df.sort_values("A_Protein", ascending=False)["a_std"])[160:210],
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
    labl = df.sort_values("A_Protein", ascending=False)["a_in_CI"][160:210].to_numpy()
    color = ["tab:orange" if item == 1 else "tab:red" for item in labl]
    ax.scatter(
        x,
        (df.sort_values("A_Protein", ascending=False)["A_Protein"])[160:210],
        s=20,
        label="Ground truth",
        color=color,
        zorder=10,
    )
    # ax.scatter(x,(df.sort_values('A_Protein',ascending=False)['a_MOM'])[160:210],s=50,label='MOM estimator',c='#4f7942',zorder=13)
    ax.legend(frameon=False, fontsize=16, markerscale=1.5)
    ax.axes.get_xaxis().set_visible(False)
    plt.xlabel("Construct Number")
    plt.ylabel("Estimated mean protein copy number", fontsize=16)
    plt.title("Confidence Intervals MLE")
    plt.savefig(
        Path(output_path) / "a_confidence_interval.png",
        transparent=True,
        bbox_inches="tight",
        dpi=600,
    )

    # fig_dims = (20, 8)
    # x = np.linspace(0, 2, 50)
    # fig, ax = plt.subplots(figsize=fig_dims)
    # ax.errorbar(x,(df.sort_values('B_Protein',ascending=False)['b_MLE'])[160:210], yerr=2*(df.sort_values('B_Protein',ascending=False)['a_std'])[160:210],  fmt=' ',color='tab:brown', marker='o', markersize=7,
    #              ecolor='gray', elinewidth=2.5, capsize=4,label='ML estimator',zorder=1)
    # labl=df.sort_values('B_Protein',ascending=False)['b_in_CI'][160:210].to_numpy()
    # color= ['tab:orange' if l == 1 else 'tab:red' for l in labl]
    # ax.scatter(x,(df.sort_values('B_Protein',ascending=False)['B_Protein'])[160:210],s=20,label='Ground truth',color=color,zorder=10)
    # # ax.scatter(x,(df.sort_values('B_Protein',ascending=False)['b_MOM'])[160:210],s=50,label='MOM estimator',c='#4f7942',zorder=13)
    # ax.legend(frameon=False,fontsize=16,markerscale=1.5)
    # ax.axes.get_xaxis().set_visible(False)
    # plt.xlabel('Construct Number')
    # plt.ylabel('Estimated std protein copy number',fontsize=16)
    # plt.title('Confidence Intervals MLE')


if __name__ == "__main__":
    main()
