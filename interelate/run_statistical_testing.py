"""Runs global & pairwise statistical testing on each query contingency table."""

from dataclasses import dataclass
from itertools import combinations
from typing import Annotated
from typing import Literal
from typing import TypeAlias
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.stats import MonteCarloMethod
from scipy.stats import PermutationMethod
from scipy.stats.contingency import association
from scipy.stats.contingency import chi2_contingency
from scipy.stats.contingency import Chi2ContingencyResult
from statsmodels.stats.multitest import multipletests

from interelate.build_contingency_tables import ContingencyTable
from interelate.build_contingency_tables import ContingencyTables


class RawGlobalTestingResult(TypedDict):
    """Dictionary shape mapping the raw global testing result for one query
      contingency table. The length of reference_names is at least two and it
      matches the columns in observed_frequencies. reference_names are
      santisised and unique."""

    reference_names: tuple[str, ...]
    overlap_statuses: tuple[Literal['overlap'], Literal['no_overlap']]
    overlap_rates: NDArray[np.float64]
    observed_frequencies: NDArray[np.int_]
    association_test_result: float | None
    chi2_test_result: Chi2ContingencyResult | None
    reject_null: bool | None


class RawPairwiseTestingResult(TypedDict):
    """Dictionary shape mapping the raw pairwise testing result for one query
      contingency table. reference_names are santisised and unique."""

    reference_names: tuple[str, str]
    overlap_statuses: tuple[Literal['overlap'], Literal['no_overlap']]
    association_test_result: float | None
    chi2_test_result: Chi2ContingencyResult | None
    adjusted_p_value: np.float64 | None
    reject_null: np.bool_ | None


class RawStatisticalResult(TypedDict):
    """Dictionary shape mapping the raw statistical results for one query
      contingency table."""

    raw_global_testing_result: RawGlobalTestingResult
    raw_pairwise_testing_results: list[RawPairwiseTestingResult] | None


RawStatisticalResults: TypeAlias = Annotated[
    dict[str, dict[str, RawStatisticalResult]],
    'Defines mapping of raw statistical results for each query contingency table. '
    'The first key is the corresponding query BED filename. The second key is the '
    'genomic distance with "bp" suffix. The value is a RawStatisticalResult '
    'dictionary. There is at least one query and one genomic distance. Filenames '
    'are santisised and unique.'
]

AssociationStatistic: TypeAlias = Annotated[
    Literal['cramer', 'tschuprow', 'pearson'],
    'Defines the association statistic options to calculate for association tests.'
]

AdjustMethod: TypeAlias = Annotated[
    Literal[
        'bonferroni',
        'sidak',
        'holm-sidak',
        'holm',
        'simes-hochberg',
        'hommel',
        'fdr_bh',
        'fdr_by',
        'fdr_tsbh',
        'fdr_tsbky',
    ],
    'Defines the adjustment method options for multiple testing correction of '
    'pairwise chi2 test p-values.'
]


@dataclass(frozen=True, slots=True, kw_only=True)
class StatisticalTestingConfig:
    """Stores the config options for global and pairwise chi2 and
    association tests on each query contingency table.

    Attributes:
        significance_level: Significance level to use for chi2 tests. Values are
          already restricted to the range [0, 1].
        yates_correction: Decision to use Yates correction for chi2 and
          association tests. This is forced to be False in global chi2 tests if
          there are more than two reference BED files.
        power_divergence_lambda: If given, the statistic to use from the
          Cressie-Read power divergence family in place of Pearsons chi2 statistic
          in chi2 and association tests.
        resampling_method: If given, the resampling method to use for chi2
          tests. An error is raised if power_divergence_lambda is not None
          and/or if yates_correction is True.
        association_statistic: Statistic to calculate for association tests.
        pairwise_testing: Decision to run pairwise tests between reference BED
          files. This is forced to be False if there are only two reference BED
          files and is overridden by a non-significant global chi2 test result.
        adjust_method: Adjustment method for multiple testing correction of
          pairwise chi2 test p-values.
        adjust_max_iterations: Maximum number of iterations to perform if
          using two-stage FDR adjustment methods. -1 corresponds to full
          iterations, which is equal to the number of pairwise tests performed. 0
          uses only a single-stage FDR adjustment using a bh or bky prior fraction
          of assumed true hypotheses. This is ignored if the adjust_method is not
          fdr_tsbh or fdr_tsbky.
    """

    significance_level: float
    yates_correction: bool
    power_divergence_lambda: float | None
    resampling_method: PermutationMethod | MonteCarloMethod | None
    association_statistic: AssociationStatistic
    adjust_method: AdjustMethod
    pairwise_testing: bool
    adjust_max_iterations: int


def run_query_global_testing(
    query_contingency_table: ContingencyTable,
    statistical_testing_config: StatisticalTestingConfig,
) -> RawGlobalTestingResult:
    """Runs global testing for one query contingency table."""

    observed_frequencies = query_contingency_table.observed_frequencies

    overlap_rates = observed_frequencies[0] / observed_frequencies.sum(axis=0)

    if np.any(observed_frequencies.sum(axis=1) == 0):
        association_test_result = None
        chi2_test_result = None
        reject_null = None

    else:
        yates_correction = (
            statistical_testing_config.yates_correction
            and len(query_contingency_table.reference_names) == 2
        )

        association_test_result = association(
            observed_frequencies,
            method=statistical_testing_config.association_statistic,
            correction=yates_correction,
            lambda_=statistical_testing_config.power_divergence_lambda
        )

        chi2_test_result = chi2_contingency(
            observed_frequencies,
            correction=yates_correction,
            lambda_=statistical_testing_config.power_divergence_lambda,
            method=statistical_testing_config.resampling_method
        )

        reject_null = bool(
            chi2_test_result.pvalue
            <= statistical_testing_config.significance_level
        )

    raw_global_testing_result = {
        'reference_names': query_contingency_table.reference_names,
        'overlap_statuses': query_contingency_table.overlap_statuses,
        'overlap_rates': overlap_rates,
        'observed_frequencies': observed_frequencies,
        'association_test_result': association_test_result,
        'chi2_test_result': chi2_test_result,
        'reject_null': reject_null
    }

    return raw_global_testing_result


def run_query_pairwise_testing(
    query_contingency_table: ContingencyTable,
    statistical_testing_config: StatisticalTestingConfig,
) -> list[RawPairwiseTestingResult]:
    """Runs pairwise testing for one query contingency table."""

    reference_names = query_contingency_table.reference_names
    reference_pairs = tuple(combinations(range(len(reference_names)), 2))

    observed_frequencies = query_contingency_table.observed_frequencies

    raw_pairwise_testing_results = []
    valid_result_indices = []
    p_values = []

    for r1, r2 in reference_pairs:
        pairwise_observed_frequencies = observed_frequencies[:, [r1, r2]]

        if np.any(pairwise_observed_frequencies.sum(axis=1) == 0):
            association_test_result = None
            chi2_test_result = None
            p_values.append(1.0)

        else:
            association_test_result = association(
                pairwise_observed_frequencies,
                method=statistical_testing_config.association_statistic,
                correction=statistical_testing_config.yates_correction,
                lambda_=statistical_testing_config.power_divergence_lambda
            )

            chi2_test_result = chi2_contingency(
                pairwise_observed_frequencies,
                correction=statistical_testing_config.yates_correction,
                lambda_=statistical_testing_config.power_divergence_lambda,
                method=statistical_testing_config.resampling_method
            )

            valid_result_indices.append(len(raw_pairwise_testing_results))
            p_values.append(chi2_test_result.pvalue)

        raw_pairwise_testing_results.append(
            {
                'reference_names': (
                    reference_names[r1],
                    reference_names[r2]
                ),
                'overlap_statuses': query_contingency_table.overlap_statuses,
                'association_test_result': association_test_result,
                'chi2_test_result': chi2_test_result,
                'adjusted_p_value': None,
                'reject_null': None
            }
        )

    reject_nulls, adjusted_p_values, _, _ = multipletests(
        p_values,
        alpha=statistical_testing_config.significance_level,
        method=statistical_testing_config.adjust_method,
        maxiter=statistical_testing_config.adjust_max_iterations
    )

    for result_index in valid_result_indices:
        raw_pairwise_testing_results[
            result_index
        ]['adjusted_p_value'] = adjusted_p_values[result_index]

        raw_pairwise_testing_results[
            result_index
        ]['reject_null'] = reject_nulls[result_index]

    return raw_pairwise_testing_results


def run_query_statistical_testing(
    query_contingency_table: ContingencyTable,
    statistical_testing_config: StatisticalTestingConfig
) -> RawStatisticalResult:
    """Runs statistical testing on one query contingency table."""

    raw_global_testing_result = run_query_global_testing(
        query_contingency_table=query_contingency_table,
        statistical_testing_config=statistical_testing_config
    )

    chi2_test_result = raw_global_testing_result['chi2_test_result']

    if (
        chi2_test_result is not None
        and chi2_test_result.pvalue
        <= statistical_testing_config.significance_level
        and statistical_testing_config.pairwise_testing
    ):
        raw_pairwise_testing_results = run_query_pairwise_testing(
            query_contingency_table=query_contingency_table,
            statistical_testing_config=statistical_testing_config
        )
    else:
        raw_pairwise_testing_results = None

    query_raw_statistical_result = {
        'raw_global_testing_result': raw_global_testing_result,
        'raw_pairwise_testing_results': raw_pairwise_testing_results
    }

    return query_raw_statistical_result


def run_statistical_testing(
    contingency_tables: ContingencyTables,
    statistical_testing_config: StatisticalTestingConfig
) -> RawStatisticalResults:
    """Runs statistical testing on each query contingency table.

    Args:
        contingency_tables: Dictionary containing contingency tables for each
          query BED file at each genomic distance in ContingencyTables format.
        statistical_testing_config: Instance of StatisticalTestingConfig
          containing the config parameters for running all statistical testing.

    Returns:
        Dictionary containing the raw statistical results for each query
          contingency table in RawStatisticalResults format.
    """

    raw_statistical_results = {}

    for query_name, query_contingency_tables in contingency_tables.items():
        raw_statistical_results[query_name] = {}

        for overlap_name, query_contingency_table in (
            query_contingency_tables.items()
        ):
            raw_statistical_results[query_name][overlap_name] = (
                run_query_statistical_testing(
                    query_contingency_table=query_contingency_table,
                    statistical_testing_config=statistical_testing_config,
                )
            )

    return raw_statistical_results
