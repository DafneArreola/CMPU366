#!/usr/bin/env python3

from pathlib import Path

import click

from tokenizer import tokenize


def print_concordance(words: list[str], index: int, label: str = ""):
    """Print token at index with surrounding context."""

    pre_context = " ".join(words[max(0, index - 20) : index])[-30:]
    center = words[index]
    post_context = " ".join(words[index + 1 :])[:30]

    print(
        f"{pre_context:>30}  {center:<20}  {post_context:<30}  "
        f"({label:2} {index:6})"
    )


def evaluate(tokens: list[str], reference_file, hypothesis_file, verbose=0):
    """Compare hypothesis boundaries to reference and report metrics."""

    # Read boundary indices
    reference = {int(line.rstrip()) for line in reference_file}
    hypothesis = {int(line.rstrip()) for line in hypothesis_file}
    all_indices = set(range(len(tokens)))

    # Calculate classification metrics
    true_positives = hypothesis & reference
    true_negatives = all_indices - (hypothesis | reference)
    false_positives = hypothesis - reference
    false_negatives = reference - hypothesis

    # Calculate performance metrics
    precision = len(true_positives) / len(hypothesis) if hypothesis else 0.0
    recall = len(true_positives) / len(reference) if reference else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Show concordance if requested
    if verbose >= 1:
        if verbose == 2:
            # Show all predictions
            for i in sorted(true_positives):
                print_concordance(tokens, i, "TP")
            for i in sorted(true_negatives):
                print_concordance(tokens, i, "TN")
        else:
            # Show only errors
            for i in sorted(false_positives):
                print_concordance(tokens, i, "FP")
            for i in sorted(false_negatives):
                print_concordance(tokens, i, "FN")

    # Print results summary
    print(f"TP: {len(true_positives):7}\tFN: {len(false_negatives):7}")
    print(f"FP: {len(false_positives):7}\tTN: {len(true_negatives):7}")
    print()
    print(
        f"PRECISION: {precision:5.2%}\t"
        f"RECALL: {recall:5.2%}\t"
        f"F: {f1:5.2%}"
    )


@click.command()
@click.option(
    "-v",
    "--verbosity",
    type=click.IntRange(0, 2),
    default=0,
    help="0=summary, 1=show errors, 2=show all predictions.",
)
@click.option(
    "-d",
    "--data-location",
    default="brown/",
    help="Directory containing text and reference files.",
)
@click.option(
    "-c",
    "--category",
    required=True,
    help="Category name (e.g., 'editorial', 'fiction').",
)
@click.option(
    "-y",
    "--hypothesis",
    type=click.File("r"),
    required=True,
    help="File containing hypothesis boundary indices.",
)
def main(verbosity, data_location, category, hypothesis):
    """Evaluate sentence segmentation performance."""
    text_path = Path(data_location) / f"{category}.txt"
    reference_path = Path(data_location) / f"{category}-eos.txt"

    # Check files exist
    if not text_path.exists():
        click.echo(f"Error: Text file not found: {text_path}", err=True)
        raise click.Abort()
    if not reference_path.exists():
        click.echo(
            f"Error: Reference file not found: {reference_path}", err=True
        )
        raise click.Abort()

    # Read and tokenize text
    with open(text_path) as f:
        text = f.read()
    tokens = tokenize(text)

    # Run evaluation
    with open(reference_path) as ref_file:
        evaluate(tokens, ref_file, hypothesis, verbosity)


if __name__ == "__main__":
    main()
