from typing import Literal
from typing import TypeAlias
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.stats.contingency import Chi2ContingencyResult

from interelate import format_statistical_results
from interelate.format_statistical_results import FormattedGlobalTestingResult
from interelate.format_statistical_results import (
    FormattedPairwiseTestingResult
)
from interelate.format_statistical_results import FormattedStatisticalResult
from interelate.run_statistical_testing import RawGlobalTestingResult
from interelate.run_statistical_testing import RawPairwiseTestingResult
from interelate.run_statistical_testing import RawStatisticalResult


GlobalOutcome: TypeAlias = Literal[
    'significant',
    'non_significant',
    'invalid'
]
PairwiseOutcome: TypeAlias = Literal['all_valid', 'mixed']


def make_raw_global_result(
    outcome: GlobalOutcome = 'significant'
) -> RawGlobalTestingResult:
    observed_frequencies = np.array([[8, 2], [2, 8]], dtype=np.int_)
    chi2_test_result = {
        'significant': Chi2ContingencyResult(
            np.float64(7.2),
            np.float64(0.01),
            np.int64(1),
            np.array([[5.0, 5.0], [5.0, 5.0]])
        ),
        'non_significant': Chi2ContingencyResult(
            np.float64(0.2),
            np.float64(0.65),
            np.int64(1),
            np.array([[5.0, 5.0], [5.0, 5.0]])
        ),
        'invalid': None
    }[outcome]
    association_test_result = {
        'significant': 0.6,
        'non_significant': 0.1,
        'invalid': None
    }[outcome]
    reject_null = {
        'significant': True,
        'non_significant': False,
        'invalid': None
    }[outcome]

    raw_global_testing_result = {
        'reference_names': ('reference_a', 'reference_b'),
        'overlap_statuses': ('overlap', 'no_overlap'),
        'overlap_rates': np.array([0.8, 0.2]),
        'observed_frequencies': observed_frequencies,
        'association_test_result': association_test_result,
        'chi2_test_result': chi2_test_result,
        'reject_null': reject_null
    }

    return raw_global_testing_result


def make_raw_pairwise_results(
    outcome: PairwiseOutcome = 'mixed'
) -> list[RawPairwiseTestingResult]:
    first_result = (
        {
            'reference_names': ('reference_a', 'reference_b'),
            'overlap_statuses': ('overlap', 'no_overlap'),
            'association_test_result': 0.1,
            'chi2_test_result': Chi2ContingencyResult(
                np.float64(0.5),
                np.float64(0.48),
                np.int64(1),
                np.array([[4.5, 5.5], [5.5, 4.5]])
            ),
            'adjusted_p_value': np.float64(0.5),
            'reject_null': np.bool_(False)
        }
        if outcome == 'all_valid'
        else {
            'reference_names': ('reference_a', 'reference_b'),
            'overlap_statuses': ('overlap', 'no_overlap'),
            'association_test_result': None,
            'chi2_test_result': None,
            'adjusted_p_value': None,
            'reject_null': None
        }
    )

    raw_pairwise_testing_results = [
        first_result,
        {
            'reference_names': ('reference_a', 'reference_c'),
            'overlap_statuses': ('overlap', 'no_overlap'),
            'association_test_result': 0.5,
            'chi2_test_result': Chi2ContingencyResult(
                np.float64(6.0),
                np.float64(0.01),
                np.int64(1),
                np.array([[2.5, 2.5], [7.5, 7.5]])
            ),
            'adjusted_p_value': np.float64(0.03),
            'reject_null': np.bool_(True)
        },
        {
            'reference_names': ('reference_b', 'reference_c'),
            'overlap_statuses': ('overlap', 'no_overlap'),
            'association_test_result': 0.25,
            'chi2_test_result': Chi2ContingencyResult(
                np.float64(2.0),
                np.float64(0.15),
                np.int64(1),
                np.array([[4.0, 6.0], [6.0, 4.0]])
            ),
            'adjusted_p_value': np.float64(0.3),
            'reject_null': np.bool_(False)
        }
    ]

    return raw_pairwise_testing_results


def make_raw_statistical_result(
    global_outcome: GlobalOutcome = 'significant',
    pairwise_outcome: PairwiseOutcome | None = None
) -> RawStatisticalResult:
    raw_statistical_result = {
        'raw_global_testing_result': make_raw_global_result(
            outcome=global_outcome
        ),
        'raw_pairwise_testing_results': (
            make_raw_pairwise_results(outcome=pairwise_outcome)
            if pairwise_outcome is not None
            else None
        )
    }

    return raw_statistical_result


def make_expected_formatted_global_result(
    outcome: GlobalOutcome = 'significant'
) -> FormattedGlobalTestingResult:
    global_testing_result = {
        'significant': {
            'expected_frequencies': {
                'overlap': [
                    ('reference_a', 5.0),
                    ('reference_b', 5.0)
                ],
                'no_overlap': [
                    ('reference_a', 5.0),
                    ('reference_b', 5.0)
                ]
            },
            'chi2_statistic': 7.2,
            'dof': 1,
            'p_value': 0.01,
            'reject_null': True,
            'association_statistic': 0.6
        },
        'non_significant': {
            'expected_frequencies': {
                'overlap': [
                    ('reference_a', 5.0),
                    ('reference_b', 5.0)
                ],
                'no_overlap': [
                    ('reference_a', 5.0),
                    ('reference_b', 5.0)
                ]
            },
            'chi2_statistic': 0.2,
            'dof': 1,
            'p_value': 0.65,
            'reject_null': False,
            'association_statistic': 0.1
        },
        'invalid': None
    }[outcome]

    formatted_global_testing_result = {
        'overlap_result': {
            'observed_frequencies': {
                'overlap': [('reference_a', 8), ('reference_b', 2)],
                'no_overlap': [('reference_a', 2), ('reference_b', 8)]
            },
            'overlap_rate': [
                ('reference_a', 0.8),
                ('reference_b', 0.2)
            ]
        },
        'global_testing_result': global_testing_result
    }

    return formatted_global_testing_result


def make_expected_formatted_pairwise_result(
    outcome: PairwiseOutcome = 'mixed'
) -> FormattedPairwiseTestingResult:
    first_expected_frequencies = (
        {
            'overlap': [('reference_a', 4.5), ('reference_b', 5.5)],
            'no_overlap': [('reference_a', 5.5), ('reference_b', 4.5)]
        }
        if outcome == 'all_valid'
        else {
            'overlap': [('reference_a', None), ('reference_b', None)],
            'no_overlap': [('reference_a', None), ('reference_b', None)]
        }
    )
    first_chi2_statistic = 0.5 if outcome == 'all_valid' else None
    first_p_value = 0.48 if outcome == 'all_valid' else None
    first_adjusted_p_value = 0.5 if outcome == 'all_valid' else None
    first_reject_null = False if outcome == 'all_valid' else None
    first_association_statistic = 0.1 if outcome == 'all_valid' else None

    formatted_pairwise_testing_result = {
        'expected_frequencies': {
            'overlap': [
                first_expected_frequencies['overlap'],
                [('reference_a', 2.5), ('reference_c', 2.5)],
                [('reference_b', 4.0), ('reference_c', 6.0)]
            ],
            'no_overlap': [
                first_expected_frequencies['no_overlap'],
                [('reference_a', 7.5), ('reference_c', 7.5)],
                [('reference_b', 6.0), ('reference_c', 4.0)]
            ]
        },
        'chi2_statistic': [
            (('reference_a', 'reference_b'), first_chi2_statistic),
            (('reference_a', 'reference_c'), 6.0),
            (('reference_b', 'reference_c'), 2.0)
        ],
        'p_value': [
            (('reference_a', 'reference_b'), first_p_value),
            (('reference_a', 'reference_c'), 0.01),
            (('reference_b', 'reference_c'), 0.15)
        ],
        'adjusted_p_value': [
            (('reference_a', 'reference_b'), first_adjusted_p_value),
            (('reference_a', 'reference_c'), 0.03),
            (('reference_b', 'reference_c'), 0.3)
        ],
        'reject_null': [
            (('reference_a', 'reference_b'), first_reject_null),
            (('reference_a', 'reference_c'), True),
            (('reference_b', 'reference_c'), False)
        ],
        'association_statistic': [
            (('reference_a', 'reference_b'), first_association_statistic),
            (('reference_a', 'reference_c'), 0.5),
            (('reference_b', 'reference_c'), 0.25)
        ]
    }

    return formatted_pairwise_testing_result


def make_expected_formatted_statistical_result(
    global_outcome: GlobalOutcome = 'significant',
    pairwise_outcome: PairwiseOutcome | None = None
) -> FormattedStatisticalResult:
    formatted_global_result = make_expected_formatted_global_result(
        outcome=global_outcome
    )
    formatted_statistical_result = {
        'overlap_result': formatted_global_result['overlap_result'],
        'global_testing_result': formatted_global_result[
            'global_testing_result'
        ],
        'pairwise_testing_result': (
            make_expected_formatted_pairwise_result(
                outcome=pairwise_outcome
            )
            if pairwise_outcome is not None
            else None
        )
    }

    return formatted_statistical_result


@pytest.mark.parametrize(
    'outcome',
    ['significant', 'non_significant'],
    ids=['significant', 'non-significant']
)
def test_format_global_testing_result_formats_valid_results(
    outcome: GlobalOutcome
) -> None:
    result = format_statistical_results.format_global_testing_result(
        make_raw_global_result(outcome=outcome)
    )

    assert result == make_expected_formatted_global_result(
        outcome=outcome
    )

    overlap_result = result['overlap_result']
    assert all(
        type(value) is int
        for values in overlap_result['observed_frequencies'].values()
        for _, value in values
    )
    assert all(
        type(value) is float
        for _, value in overlap_result['overlap_rate']
    )

    global_result = result['global_testing_result']
    assert global_result is not None
    assert type(global_result['chi2_statistic']) is float
    assert type(global_result['dof']) is int
    assert type(global_result['p_value']) is float
    assert type(global_result['reject_null']) is bool
    assert type(global_result['association_statistic']) is float
    assert all(
        type(value) is float
        for values in global_result['expected_frequencies'].values()
        for _, value in values
    )


def test_format_global_testing_result_keeps_overlap_data_when_invalid(
) -> None:
    result = format_statistical_results.format_global_testing_result(
        make_raw_global_result(outcome='invalid')
    )

    assert result == make_expected_formatted_global_result(
        outcome='invalid'
    )


@pytest.mark.parametrize(
    'outcome',
    ['all_valid', 'mixed'],
    ids=['all-valid', 'mixed-valid-invalid']
)
def test_format_pairwise_testing_result_formats_structural_outcomes(
    outcome: PairwiseOutcome
) -> None:
    result = format_statistical_results.format_pairwise_testing_result(
        make_raw_pairwise_results(outcome=outcome)
    )

    assert result == make_expected_formatted_pairwise_result(
        outcome=outcome
    )

    for result_name in (
        'chi2_statistic',
        'p_value',
        'adjusted_p_value',
        'association_statistic'
    ):
        assert all(
            type(value) is float
            for _, value in result[result_name]
            if value is not None
        )

    assert all(
        type(value) is bool
        for _, value in result['reject_null']
        if value is not None
    )
    assert all(
        type(value) is float
        for values_by_comparison in result['expected_frequencies'].values()
        for comparison_values in values_by_comparison
        for _, value in comparison_values
        if value is not None
    )


@pytest.mark.parametrize(
    ('global_outcome', 'pairwise_outcome'),
    [
        ('significant', 'all_valid'),
        ('significant', 'mixed'),
        ('significant', None),
        ('non_significant', None),
        ('invalid', None)
    ],
    ids=[
        'significant-all-pairs-valid',
        'significant-mixed-pairs',
        'significant-pairwise-disabled',
        'non-significant',
        'invalid-global'
    ]
)
def test_format_query_statistical_result_formats_structural_outcomes(
    global_outcome: GlobalOutcome,
    pairwise_outcome: PairwiseOutcome | None
) -> None:
    raw_result = make_raw_statistical_result(
        global_outcome=global_outcome,
        pairwise_outcome=pairwise_outcome
    )

    result = format_statistical_results.format_query_statistical_result(
        raw_result
    )

    assert result == make_expected_formatted_statistical_result(
        global_outcome=global_outcome,
        pairwise_outcome=pairwise_outcome
    )


def test_format_query_statistical_result_combines_global_and_pairwise_data(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_result = make_raw_statistical_result(pairwise_outcome='mixed')
    formatted_global_result = make_expected_formatted_global_result()
    formatted_pairwise_result = make_expected_formatted_pairwise_result()
    format_global = MagicMock(return_value=formatted_global_result)
    format_pairwise = MagicMock(return_value=formatted_pairwise_result)
    monkeypatch.setattr(
        format_statistical_results,
        'format_global_testing_result',
        format_global
    )
    monkeypatch.setattr(
        format_statistical_results,
        'format_pairwise_testing_result',
        format_pairwise
    )

    result = format_statistical_results.format_query_statistical_result(
        raw_result
    )

    format_global.assert_called_once_with(
        raw_global_testing_result=raw_result['raw_global_testing_result']
    )
    format_pairwise.assert_called_once_with(
        raw_pairwise_testing_results=raw_result[
            'raw_pairwise_testing_results'
        ]
    )
    assert result['overlap_result'] is formatted_global_result[
        'overlap_result'
    ]
    assert result['global_testing_result'] is formatted_global_result[
        'global_testing_result'
    ]
    assert result['pairwise_testing_result'] is formatted_pairwise_result


def test_format_query_statistical_result_preserves_absent_pairwise_results(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_result = make_raw_statistical_result()
    formatted_global_result = make_expected_formatted_global_result()
    format_global = MagicMock(return_value=formatted_global_result)
    format_pairwise = MagicMock()
    monkeypatch.setattr(
        format_statistical_results,
        'format_global_testing_result',
        format_global
    )
    monkeypatch.setattr(
        format_statistical_results,
        'format_pairwise_testing_result',
        format_pairwise
    )

    result = format_statistical_results.format_query_statistical_result(
        raw_result
    )

    format_global.assert_called_once_with(
        raw_global_testing_result=raw_result['raw_global_testing_result']
    )
    format_pairwise.assert_not_called()
    assert result['overlap_result'] is formatted_global_result[
        'overlap_result'
    ]
    assert result['global_testing_result'] is formatted_global_result[
        'global_testing_result'
    ]
    assert result['pairwise_testing_result'] is None


def test_format_statistical_results_preserves_query_and_distance_names(
) -> None:
    raw_a = make_raw_statistical_result()
    raw_b = make_raw_statistical_result(
        global_outcome='non_significant'
    )
    raw_c = make_raw_statistical_result(pairwise_outcome='mixed')
    raw_results = {
        'query_one': {'0bp': raw_a, '10bp': raw_b},
        'query_two': {'0bp': raw_c}
    }

    result = format_statistical_results.format_statistical_results(
        raw_results
    )

    assert result == {
        'query_one': {
            '0bp': make_expected_formatted_statistical_result(),
            '10bp': make_expected_formatted_statistical_result(
                global_outcome='non_significant'
            )
        },
        'query_two': {
            '0bp': make_expected_formatted_statistical_result(
                pairwise_outcome='mixed'
            )
        }
    }


def test_format_statistical_results_delegates_every_query_distance_pair(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_a = make_raw_statistical_result()
    raw_b = make_raw_statistical_result(global_outcome='invalid')
    raw_c = make_raw_statistical_result(pairwise_outcome='mixed')
    raw_results = {
        'query_one': {'0bp': raw_a, '10bp': raw_b},
        'query_two': {'0bp': raw_c}
    }
    formatted_outputs = (
        make_expected_formatted_statistical_result(),
        make_expected_formatted_statistical_result(
            global_outcome='invalid'
        ),
        make_expected_formatted_statistical_result(
            pairwise_outcome='mixed'
        )
    )
    calls = []

    def fake_format(
        query_raw_statistical_result: RawStatisticalResult
    ) -> FormattedStatisticalResult:
        calls.append(query_raw_statistical_result)
        return formatted_outputs[len(calls) - 1]

    monkeypatch.setattr(
        format_statistical_results,
        'format_query_statistical_result',
        fake_format
    )

    result = format_statistical_results.format_statistical_results(
        raw_results
    )

    assert calls == [raw_a, raw_b, raw_c]
    assert result['query_one']['0bp'] is formatted_outputs[0]
    assert result['query_one']['10bp'] is formatted_outputs[1]
    assert result['query_two']['0bp'] is formatted_outputs[2]
