"""Parses command-line arguments for the Interelate pipeline."""

from dataclasses import replace
import logging
from pathlib import Path
import re
import sys
from typing import cast

from absl import app
from absl import flags
from scipy.stats import MonteCarloMethod
from scipy.stats import PermutationMethod

from interelate.run_pipeline import run_pipeline
from interelate.load_beds import load_beds
from interelate.run_statistical_testing import AdjustMethod
from interelate.run_statistical_testing import AssociationStatistic
from interelate.run_statistical_testing import StatisticalTestingConfig
from interelate.write_output import RUN_LOG_HANDLER


# I/O paths
REFERENCE_DIR = flags.DEFINE_string(
    'reference_dir',
    None,
    'Path to the directory containing reference BED files.'
)
QUERY_DIR = flags.DEFINE_string(
    'query_dir',
    None,
    'Path to the directory containing query BED files.'
)
OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'Path to the directory where all output files will be written.'
)

# Distances for counting genomic interval overlaps
GENOMIC_DISTANCES = flags.DEFINE_list(
    'genomic_distances',
    None,
    'Comma-separated list of genomic distances (in bp) to use for counting ' 
    'overlaps between reference and query BED files. Please provide a unique list ' 
    'of non-negative integers only.' 
)

# Statistical testing config
SIGNIFICANCE_LEVEL = flags.DEFINE_float(
    'significance_level',
    0.05,
    'Significance level to use for chi2 square tests.',
    lower_bound=0,
    upper_bound=1
)
YATES_CORRECTION = flags.DEFINE_boolean(
    'yates_correction',
    False,
    'Whether to apply Yates\' correction for chi2 and association tests.'
)
POWER_DIVERGENCE_LAMBDA = flags.DEFINE_float(
    'power_divergence_lambda',
    None,
    'Statistic to use from the Cressie-Read power divergence family in place of ' 
    'Pearson\'s chi2 statistic in chi2 and association tests.'
)
RESAMPLING_METHOD = flags.DEFINE_enum(
    'resampling_method',
    None,
    ['permutation', 'monte_carlo'],
    'Resampling method to use for chi2 tests.'
)
ASSOCIATION_STATISTIC = flags.DEFINE_enum(
    'association_statistic',
    'cramer',
    ['cramer', 'tschuprow', 'pearson'],
    'Statistic to calculate for association tests.'
)
PAIRWISE_TESTING = flags.DEFINE_boolean(
    'pairwise_testing',
    True,
    'Whether to run pairwise tests between reference BED files.'
)
ADJUST_METHOD = flags.DEFINE_enum(
    'adjust_method',
    'holm-sidak',
    [
        'bonferroni',
        'sidak',
        'holm-sidak',
        'holm',
        'simes-hochberg',
        'hommel',
        'fdr_bh',
        'fdr_by',
        'fdr_tsbh',
        'fdr_tsbky'
    ],
    'Adjustment method for multiple testing correction of pairwise chi2 test ' 
    'p-values.'
)
ADJUST_MAX_ITERATIONS = flags.DEFINE_integer(
    'adjust_max_iterations',
    1,
    'Maximum number of iterations to perform when using two-stage FDR adjustment ' 
    'methods. -1 corresponds to full iterations, which is equal to the number of '
    'pairwise tests performed. 0 uses only a single-stage FDR adjustment using a '
    'bh or bky prior fraction of assumed true hypotheses.'
)


def main(_):
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            RUN_LOG_HANDLER
        ],
        force=True
    )

    # Fail early on statistical testing config discrepancies
    if RESAMPLING_METHOD.value is not None:
        if YATES_CORRECTION.value:
            raise ValueError(
                '--resampling_method must be None if --yates_correction is True.'
            )

        if POWER_DIVERGENCE_LAMBDA.value is not None:
            raise ValueError(
                '--resampling_method must be None if --power_divergence_lambda is not None.'
            )
    
    resampling_method = (
        PermutationMethod()
        if RESAMPLING_METHOD.value == 'permutation'
        else MonteCarloMethod()
        if RESAMPLING_METHOD.value == 'monte_carlo'
        else None
    )

    statistical_testing_config = StatisticalTestingConfig(
        significance_level=SIGNIFICANCE_LEVEL.value,
        yates_correction=YATES_CORRECTION.value,
        power_divergence_lambda=POWER_DIVERGENCE_LAMBDA.value,
        resampling_method=resampling_method,
        association_statistic=cast(AssociationStatistic, ASSOCIATION_STATISTIC.value),
        pairwise_testing=PAIRWISE_TESTING.value,
        adjust_method=cast(AdjustMethod, ADJUST_METHOD.value),
        adjust_max_iterations=ADJUST_MAX_ITERATIONS.value
    )

    # Fail early if no genomic distances given as non-negative integers
    genomic_distances = tuple(sorted({
        int(gd.strip())
        for gd in GENOMIC_DISTANCES.value
        if re.fullmatch(r'[0-9]+', gd.strip())
    }))

    if not genomic_distances:
        raise ValueError('--genomic_distances contains no non-negative integers.')
    
    
    logging.info(
        'The following genomic distances will be used after filtering for non-negative ' 
        f'integers: {genomic_distances}.'
    )

    # Fail early if output directory cannot be created
    output_dir = Path(OUTPUT_DIR.value)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f'Failed while creating --output_dir: {e}')
        raise
    
    # Load and validate BED files 
    reference_dir = Path(REFERENCE_DIR.value)
    query_dir = Path(QUERY_DIR.value)

    beds = load_beds(
        bed_dirs={
            'reference': reference_dir,
            'query': query_dir
        }
    )

    if len(beds['reference']) < 3 and statistical_testing_config.pairwise_testing:
        statistical_testing_config = replace(
            statistical_testing_config,
            pairwise_testing=False
        )
        logging.warning('--pairwise_testing True forced to False as < 3 references.')

    if len(beds['reference']) >= 3 and YATES_CORRECTION.value:
        logging.warning(
            '--yates_correction True ignored in global testing as >= 3 references.'
        )
    
    if (
        ADJUST_MAX_ITERATIONS.present
        and ADJUST_METHOD.value not in ('fdr_tsbh', 'fdr_tsbky')
    ):
        logging.warning(
            '--adjust_max_iterations ignored as --adjust_method is not two-stage FDR.'
        )
    
    logging.info(
        'The following statistical testing configuration will be used: '
        f'{statistical_testing_config}.'
    )

    # Run Interelate pipeline
    run_pipeline(
        beds=beds,
        output_dir=output_dir,
        genomic_distances=genomic_distances,
        statistical_testing_config=statistical_testing_config
    )


# Script entry point
def cli():
    flags.mark_flags_as_required([
        'reference_dir',
        'query_dir',
        'output_dir',
        'genomic_distances'
    ])

    app.run(main)
