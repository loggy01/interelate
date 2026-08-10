"""Formats each query raw statistical result into the final JSON-ready
  shape."""

from typing import Annotated
from typing import TypeAlias
from typing import TypedDict

from interelate.run_statistical_testing import RawGlobalTestingResult
from interelate.run_statistical_testing import RawPairwiseTestingResult
from interelate.run_statistical_testing import RawStatisticalResult
from interelate.run_statistical_testing import RawStatisticalResults


ReferenceIntValue: TypeAlias = Annotated[
    tuple[str, int],
    'Defines a reference name and integer value pair. The reference name is '
    'sanitised and unique.'
]

ReferenceFloatValue: TypeAlias = Annotated[
    tuple[str, float],
    'Defines a reference name and float value pair. The reference name is '
    'sanitised and unique.'
]

PairwiseReferenceFloatValue: TypeAlias = Annotated[
    tuple[str, float | None],
    'Defines a pairwise reference name and optional float value pair. The '
    'reference name is sanitised and unique.'
]

ComparisonFloatValue: TypeAlias = Annotated[
    tuple[tuple[str, str], float | None],
    'Defines a pairwise reference comparison and matching optional float value '
    'pair. The reference names are sanitised and unique.'
]

ComparisonBoolValue: TypeAlias = Annotated[
    tuple[tuple[str, str], bool | None],
    'Defines a pairwise reference comparison and matching optional boolean value '
    'pair. The reference names are sanitised and unique.'
]


class ObservedFrequencies(TypedDict):
    """Dictionary shape mapping the global observed frequencies for one
      query raw statistical result. Both lists match the number of
      references."""

    overlap: list[ReferenceIntValue]
    no_overlap: list[ReferenceIntValue]


class OverlapResult(TypedDict):
    """Dictionary shape mapping the overlap result for one query raw
      statistical result. The list matches the number of references."""

    observed_frequencies: ObservedFrequencies
    overlap_rate: list[ReferenceFloatValue]


class ExpectedFrequencies(TypedDict):
    """Dictionary shape mapping the global expected frequencies for one
      query raw statistical result. Both lists match the number of
      references."""

    overlap: list[ReferenceFloatValue]
    no_overlap: list[ReferenceFloatValue]


class PairwiseExpectedFrequencies(TypedDict):
    """Dictionary shape mapping the pairwise expected frequencies for one
      query raw statistical result. Both outer lists match the number of
      pairwise comparisons between references. Both inner lists are length two,
      representing the two references in the pairwise comparison."""

    overlap: list[list[PairwiseReferenceFloatValue]]
    no_overlap: list[list[PairwiseReferenceFloatValue]]


class GlobalTestingResult(TypedDict):
    """Dictionary shape mapping the global testing result for one query raw
      statistical result."""

    expected_frequencies: ExpectedFrequencies
    chi2_statistic: float
    dof: int
    p_value: float
    reject_null: bool
    association_statistic: float


class FormattedGlobalTestingResult(TypedDict):
    """Dictionary shape mapping the formatted global testing result for one
      query raw statistical result."""

    overlap_result: OverlapResult
    global_testing_result: GlobalTestingResult | None


class FormattedPairwiseTestingResult(TypedDict):
    """Dictionary shape mapping the formatted pairwise testing result for one
      query raw statistical result. The lists match the number of pairwise
      comparisons between references."""

    expected_frequencies: PairwiseExpectedFrequencies
    chi2_statistic: list[ComparisonFloatValue]
    p_value: list[ComparisonFloatValue]
    adjusted_p_value: list[ComparisonFloatValue]
    reject_null: list[ComparisonBoolValue]
    association_statistic: list[ComparisonFloatValue]


class FormattedStatisticalResult(TypedDict):
    """Dictionary shape mapping the formatted statistical result for one
      query raw statistical result."""

    overlap_result: OverlapResult
    global_testing_result: GlobalTestingResult | None
    pairwise_testing_result: FormattedPairwiseTestingResult | None


FormattedStatisticalResults: TypeAlias = Annotated[
    dict[str, dict[str, FormattedStatisticalResult]],
    'Defines mapping of formatted statistical results for each query raw '
    'statistical result. The first key is the corresponding query BED filename. '
    'The second key is the genomic distance with "bp" suffix. The value is a '
    'FormattedStatisticalResult dictionary. There is at least one query and one '
    'genomic distance. Filenames are sanitised and unique.'
]


def format_global_testing_result(
    raw_global_testing_result: RawGlobalTestingResult
) -> FormattedGlobalTestingResult:
    """Formats one query raw global testing result."""

    reference_names = raw_global_testing_result['reference_names']

    observed_frequencies = {
        f'{overlap_status}': [
            (reference_name, int(value))
            for reference_name, value in zip(reference_names, row)
        ]
        for overlap_status, row in zip(
            raw_global_testing_result['overlap_statuses'],
            raw_global_testing_result['observed_frequencies']
        )
    }

    overlap_rate = [
        (reference_name, float(value))
        for reference_name, value in zip(
            reference_names,
            raw_global_testing_result['overlap_rates']
        )
    ]

    chi2_test_result = raw_global_testing_result['chi2_test_result']

    if chi2_test_result is not None:
        expected_frequencies = {
            f'{overlap_status}': [
                (reference_name, float(value))
                for reference_name, value in zip(reference_names, row)
            ]
            for overlap_status, row in zip(
                raw_global_testing_result['overlap_statuses'],
                chi2_test_result.expected_freq
            )
        }

        global_testing_result = {
            'expected_frequencies': expected_frequencies,
            'chi2_statistic': float(chi2_test_result.statistic),
            'dof': int(chi2_test_result.dof),
            'p_value': float(chi2_test_result.pvalue),
            'reject_null': bool(
                raw_global_testing_result['reject_null']
            ),
            'association_statistic': float(
                raw_global_testing_result['association_test_result']
            )
        }

    else:
        global_testing_result = None

    formatted_global_testing_result = {
        'overlap_result': {
            'observed_frequencies': observed_frequencies,
            'overlap_rate': overlap_rate,
        },
        'global_testing_result': global_testing_result  # ALTERED
    }

    return formatted_global_testing_result


def format_pairwise_testing_result(
    raw_pairwise_testing_results: list[RawPairwiseTestingResult]
) -> FormattedPairwiseTestingResult:
    """Formats one query raw pairwise testing results."""

    formatted_pairwise_testing_result = {
        'expected_frequencies': {
            overlap_status: []
            for overlap_status in (
                raw_pairwise_testing_results[0]['overlap_statuses']
            )
        },
        'chi2_statistic': [],
        'p_value': [],
        'adjusted_p_value': [],
        'reject_null': [],
        'association_statistic': []
    }

    for raw_pairwise_testing_result in raw_pairwise_testing_results:
        comparison = raw_pairwise_testing_result['reference_names']
        chi2_test_result = raw_pairwise_testing_result['chi2_test_result']

        chi2_statistic = None
        p_value = None
        adjusted_p_value = None
        reject_null = None
        association_statistic = None

        if chi2_test_result is not None:
            chi2_statistic = float(chi2_test_result.statistic)
            p_value = float(chi2_test_result.pvalue)
            adjusted_p_value = float(raw_pairwise_testing_result['adjusted_p_value'])
            reject_null = bool(raw_pairwise_testing_result['reject_null'])
            association_statistic = float(raw_pairwise_testing_result['association_test_result'])

        for overlap_status_index, overlap_status in enumerate(
            raw_pairwise_testing_result['overlap_statuses']
        ):
            if chi2_test_result is not None:
                expected_frequencies = [
                    (reference_name, float(value))
                    for reference_name, value in zip(
                        comparison,
                        chi2_test_result.expected_freq[overlap_status_index]
                    )
                ]
            else:
                expected_frequencies = [(reference_name, None) for reference_name in comparison]

            formatted_pairwise_testing_result[
                'expected_frequencies'
            ][overlap_status].append(expected_frequencies)

        formatted_pairwise_testing_result['chi2_statistic'].append(
            (comparison, chi2_statistic)
        )

        formatted_pairwise_testing_result['p_value'].append(
            (comparison, p_value)
        )

        formatted_pairwise_testing_result['adjusted_p_value'].append(
            (comparison, adjusted_p_value)
        )

        formatted_pairwise_testing_result['reject_null'].append(
            (comparison, reject_null)
        )

        formatted_pairwise_testing_result[
            'association_statistic'
        ].append(
            (comparison, association_statistic)
        )

    return formatted_pairwise_testing_result


def format_query_statistical_result(
    query_raw_statistical_result: RawStatisticalResult
) -> FormattedStatisticalResult:
    """Formats one query raw statistical result."""

    formatted_global_testing_result = format_global_testing_result(
        raw_global_testing_result=query_raw_statistical_result[
            'raw_global_testing_result'
        ]
    )

    if query_raw_statistical_result['raw_pairwise_testing_results'] is not None:
        formatted_pairwise_testing_result = format_pairwise_testing_result(
            raw_pairwise_testing_results=query_raw_statistical_result[
                'raw_pairwise_testing_results'
            ]
        )
    else:
        formatted_pairwise_testing_result = None

    formatted_statistical_result = {
        'overlap_result': formatted_global_testing_result['overlap_result'],
        'global_testing_result': formatted_global_testing_result['global_testing_result'],
        'pairwise_testing_result': formatted_pairwise_testing_result
    }

    return formatted_statistical_result


def format_statistical_results(
    raw_statistical_results: RawStatisticalResults
) -> FormattedStatisticalResults:
    """Formats each query raw statistical result.

    Args:
        raw_statistical_results: Dictionary containing the raw statistical
          results for each query raw statistical result in RawStatisticalResults
          format.

    Returns:
        Dictionary containing the formatted statistical results for each query
          raw statistical result in FormattedStatisticalResults format.
    """

    formatted_statistical_results = {}

    for query_name, query_raw_statistical_results in (
        raw_statistical_results.items()
    ):
        formatted_statistical_results[query_name] = {}

        for overlap_name, query_raw_statistical_result in (
            query_raw_statistical_results.items()
        ):
            formatted_statistical_results[query_name][overlap_name] = (
                format_query_statistical_result(
                    query_raw_statistical_result=(
                        query_raw_statistical_result
                    )
                )
            )

    return formatted_statistical_results
