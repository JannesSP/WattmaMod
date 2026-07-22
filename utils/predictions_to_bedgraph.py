#!/usr/bin/env python3
# author: Jannes Spangenberg
# github: https://github.com/JannesSP
# website: https://jannessp.github.io

"""
Convert modification prediction calls with reference positions into a
bedGraph file suitable for conversion to bigWig.

Input format (tab-separated):
chrom  position  motif  read_id  mod_state  prediction_call

The script assumes positions are 1-based reference coordinates and converts
them to 0-based half-open BED coordinates.

Multiple predictions at the same genomic position are aggregated by mean
prediction probability.
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
import pandas as pd


def parse() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input prediction file (tab-separated)."
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output bedGraph file."
    )

    return parser.parse_args()


def main() -> None:
    args = parse()

    columns = [
        "chrom",
        "position",
        "motif",
        "read_id",
        "mod_state",
        "prediction_call"
    ]

    # Read prediction table
    df = pd.read_csv(
        args.input,
        sep="\t",
        header=None,
        names=columns,
        compression="zstd"
    )

    # Ensure numeric positions and scores
    df["position"] = pd.to_numeric(df["position"])
    df["prediction_call"] = pd.to_numeric(df["prediction_call"])

    # Apply modification threshold
    df["modified"] = df["prediction_call"] >= 0.5

    # Aggregate multiple reads covering the same reference position
    # bedgraph = (
    #     df.groupby(
    #         ["chrom", "position"],
    #         as_index=False
    #     )["prediction_call"]
    #     .mean()
    # )

    # Calculate fraction of modified reads per position
    bedgraph = (
        df.groupby(
            ["chrom", "position"],
            as_index=False
        )
        .agg(
            fraction_modified=("modified", "mean")
        )
    )

    # Convert 1-based coordinates to 0-based bedGraph coordinates
    bedgraph["start"] = bedgraph["position"] - 1
    bedgraph["end"] = bedgraph["position"]

    # Sort according to bedGraph requirements
    bedgraph = bedgraph.sort_values(
        ["chrom", "start"]
    )

    # Write bedGraph
    bedgraph[
        ["chrom", "start", "end", "fraction_modified"]
    ].to_csv(
        args.output,
        sep="\t",
        header=False,
        index=False
    )


if __name__ == '__main__':
    main()