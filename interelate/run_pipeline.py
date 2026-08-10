"""Runs Interelate pipeline on parsed command-line arguments."""

import logging
from pathlib import Path

from interelate.build_contingency_tables import build_contingency_tables
from interelate.calculate_overlap_counts import calculate_overlap_counts
from interelate.calculate_overlap_counts import GenomicDistances
from interelate.format_statistical_results import format_statistical_results
from interelate.load_beds import Beds
from interelate.run_statistical_testing import run_statistical_testing
from interelate.run_statistical_testing import StatisticalTestingConfig
from interelate.write_output import write_output


def run_pipeline(
    beds: Beds,
    genomic_distances: GenomicDistances,
    statistical_testing_config: StatisticalTestingConfig,
    output_dir: Path
) -> None:
    """Runs Interelate pipeline on parsed command-line arguments.

    Args:
        beds: Dictionary containing loaded reference and query BED files in 
          Beds format.
        genomic_distances: Tuple containing genomic distances in 
          GenomicDistances format.
        statistical_testing_config: Instance of StatisticalTestingConfig
          containing the config parameters for running all statistical testing.
        output_dir: Path to the directory where all output files will be 
          written.

    Returns:
        None.
    """

    logging.info('Counting BED file overlaps...')
    overlap_counts = calculate_overlap_counts(
        beds=beds,
        genomic_distances=genomic_distances
    )

    logging.info('Building contingency tables...')
    contingency_tables = build_contingency_tables(
        overlap_counts=overlap_counts,
        genomic_distances=genomic_distances
    )

    logging.info('Running statistical tests...')
    raw_statistical_results = run_statistical_testing(
        contingency_tables=contingency_tables,
        statistical_testing_config=statistical_testing_config
    )

    logging.info('Formatting statistical results...')
    formatted_statistical_results = format_statistical_results(
        raw_statistical_results=raw_statistical_results
    )

    logging.info('Writing output files...')
    write_output(
        overlap_counts=overlap_counts,
        formatted_statistical_results=formatted_statistical_results,
        output_dir=output_dir
    )
