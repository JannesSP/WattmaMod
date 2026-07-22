#!/usr/bin/env python3
# author: Jannes Spangenberg
# github: https://github.com/JannesSP
# website: https://jannessp.github.io

"""
Convert modification prediction calls with reference positions into a
bedMethyl file.

Input format (tab-separated):
chrom  position  motif  read_id  mod_state  prediction_call

The input file can be zstd compressed (.zst).

Positions are assumed to be 1-based reference coordinates and are converted
to 0-based half-open BED coordinates.

The output contains:
chrom
start
end
modification
score
strand
thickStart
thickEnd
itemRgb
coverage
percent_modified

The modification fraction is calculated as:

    modified reads (prediction_call >= 0.5)
    --------------------------------------
             total reads at position

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
        help="Input prediction file (tab-separated, optionally zstd compressed)."
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output bedMethyl file."
    )

    parser.add_argument(
        "-m", "--modification",
        type=str,
        required=True,
        help="Modification name (e.g. m6A, 5mC)."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prediction threshold for calling modified reads."
    )

    parser.add_argument(
        "--strand",
        type=str,
        default="+",
        help="Strand annotation."
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

    df = pd.read_csv(
        args.input,
        sep="\t",
        header=None,
        names=columns,
        compression="zstd"
    )

    df["position"] = pd.to_numeric(df["position"])
    df["prediction_call"] = pd.to_numeric(df["prediction_call"])

    # Apply modification threshold
    df["modified"] = df["prediction_call"] >= args.threshold

    # Aggregate reads per genomic position
    bedmethyl = (
        df.groupby(
            ["chrom", "position"],
            as_index=False
        )
        .agg(
            coverage=("modified", "count"),
            modified_reads=("modified", "sum")
        )
    )

    # Calculate percentage modified
    bedmethyl["percent_modified"] = (
        bedmethyl["modified_reads"]
        / bedmethyl["coverage"]
        * 100
    )

    # Convert to BED coordinates
    bedmethyl["start"] = bedmethyl["position"] - 1
    bedmethyl["end"] = bedmethyl["position"]

    # BEDMethyl columns
    bedmethyl["modification"] = args.modification

    mean_prediction = (
        df.groupby(
            ["chrom", "position"]
        )["prediction_call"]
        .mean()
        .reset_index(name="mean_prediction")
    )

    bedmethyl = bedmethyl.merge(
        mean_prediction,
        on=["chrom", "position"]
    )

    # use mean prediction for score, scaled to 0-1000 and capped at 1000 (just a safety measure, should not happen)
    bedmethyl["score"] = (
        bedmethyl["mean_prediction"] * 1000
    ).clip(
        upper=1000
    ).astype(int)

    bedmethyl["strand"] = args.strand
    bedmethyl["thickStart"] = bedmethyl["start"]
    bedmethyl["thickEnd"] = bedmethyl["end"]
    bedmethyl["itemRgb"] = 0

    # Sort output
    bedmethyl = bedmethyl.sort_values(
        ["chrom", "start"]
    )

    output = bedmethyl[
        [
            "chrom",
            "start",
            "end",
            "modification",
            "score",
            "strand",
            "thickStart",
            "thickEnd",
            "itemRgb",
            "coverage",
            "percent_modified"
        ]
    ]

    output.to_csv(
        args.output,
        sep="\t",
        header=False,
        index=False
    )


if __name__ == '__main__':
    main()