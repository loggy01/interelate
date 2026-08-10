from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.stats import PermutationMethod
from scipy.stats.contingency import Chi2ContingencyResult

from interelate import run_statistical_testing
from interelate.build_contingency_tables import ContingencyTable
from interelate.run_statistical_testing import AdjustMethod
from interelate.run_statistical_testing import AssociationStatistic
from interelate.run_statistical_testing import RawGlobalTestingResult
from interelate.run_statistical_testing import RawPairwiseTestingResult
from interelate.run_statistical_testing import RawStatisticalResult
from interelate.run_statistical_testing import StatisticalTestingConfig


def make_table(
    frequencies: list[list[int]],
    names: tuple[str, ...] = ('reference_a', 'reference_b', 'reference_c')
) -> ContingencyTable:
    contingency_table = ContingencyTable(
        reference_names=names,
        observed_frequencies=np.asarray(frequencies, dtype=np.int_)
    )

    return contingency_table


def make_raw_result(table: ContingencyTable) -> RawStatisticalResult:
    raw_statistical_result = {
        'raw_global_testing_result': {
            'reference_names': table.reference_names,
            'overlap_statuses': table.overlap_statuses,
            'overlap_rates': np.zeros(len(table.reference_names)),
            'observed_frequencies': table.observed_frequencies,
            'association_test_result': None,
            'chi2_test_result': None,
            'reject_null': None
        },
        'raw_pairwise_testing_results': None
    }

    return raw_statistical_result


def assert_float_equal(
    actual: float | np.float64,
    expected: float
) -> None:
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-300)


def assert_chi2_result(
    result: Chi2ContingencyResult | None,
    expected_statistic: float,
    expected_p_value: float,
    expected_dof: int,
    expected_frequencies: list[list[float]]
) -> None:
    assert result is not None
    assert_float_equal(result.statistic, expected_statistic)
    assert_float_equal(result.pvalue, expected_p_value)
    assert result.dof == expected_dof
    np.testing.assert_allclose(
        result.expected_freq,
        expected_frequencies,
        rtol=1e-12,
        atol=0.0
    )


def assert_global_result(
    result: RawGlobalTestingResult,
    table: ContingencyTable,
    expected_rates: list[float],
    expected_association: float,
    expected_statistic: float,
    expected_p_value: float,
    expected_frequencies: list[list[float]],
    expected_reject: bool
) -> None:
    assert result['reference_names'] == table.reference_names
    assert result['overlap_statuses'] == table.overlap_statuses
    np.testing.assert_array_equal(
        result['observed_frequencies'],
        table.observed_frequencies
    )
    np.testing.assert_allclose(
        result['overlap_rates'],
        expected_rates,
        rtol=1e-12,
        atol=0.0
    )

    association_result = result['association_test_result']
    assert association_result is not None
    assert_float_equal(association_result, expected_association)
    assert_chi2_result(
        result=result['chi2_test_result'],
        expected_statistic=expected_statistic,
        expected_p_value=expected_p_value,
        expected_dof=2,
        expected_frequencies=expected_frequencies
    )
    assert result['reject_null'] is expected_reject


def assert_pairwise_result(
    result: RawPairwiseTestingResult,
    expected_reference_names: tuple[str, str],
    expected_association: float,
    expected_statistic: float,
    expected_p_value: float,
    expected_frequencies: list[list[float]],
    expected_adjusted_p_value: float,
    expected_reject: bool
) -> None:
    assert result['reference_names'] == expected_reference_names
    assert result['overlap_statuses'] == ('overlap', 'no_overlap')
    association_result = result['association_test_result']
    assert association_result is not None
    assert_float_equal(association_result, expected_association)
    assert_chi2_result(
        result=result['chi2_test_result'],
        expected_statistic=expected_statistic,
        expected_p_value=expected_p_value,
        expected_dof=1,
        expected_frequencies=expected_frequencies
    )
    adjusted_p_value = result['adjusted_p_value']
    assert adjusted_p_value is not None
    assert_float_equal(adjusted_p_value, expected_adjusted_p_value)
    assert result['reject_null'] is not None
    assert bool(result['reject_null']) is expected_reject


def assert_standard_significant_pairwise_results(
    results: list[RawPairwiseTestingResult]
) -> None:
    expected_results = (
        (
            ('reference_a', 'reference_b'),
            0.8,
            128.0,
            1.1224297172982928e-29,
            [[50.0, 50.0], [50.0, 50.0]],
            3.3672891518948783e-29,
            True
        ),
        (
            ('reference_a', 'reference_c'),
            0.4364357804719847,
            38.095238095238095,
            6.73743601893277e-10,
            [[70.0, 70.0], [30.0, 30.0]],
            1.3474872033326236e-09,
            True
        ),
        (
            ('reference_b', 'reference_c'),
            0.4364357804719847,
            38.095238095238095,
            6.73743601893277e-10,
            [[30.0, 30.0], [70.0, 70.0]],
            1.3474872033326236e-09,
            True
        )
    )

    assert len(results) == len(expected_results)

    for result, expected in zip(results, expected_results):
        assert_pairwise_result(
            result=result,
            expected_reference_names=expected[0],
            expected_association=expected[1],
            expected_statistic=expected[2],
            expected_p_value=expected[3],
            expected_frequencies=expected[4],
            expected_adjusted_p_value=expected[5],
            expected_reject=expected[6]
        )


@pytest.mark.parametrize(
    (
        'frequencies',
        'expected_rates',
        'expected_association',
        'expected_statistic',
        'expected_p_value',
        'expected_reject'
    ),
    [
        (
            [[90, 10, 50], [10, 90, 50]],
            [0.9, 0.1, 0.5],
            0.6531972647421809,
            128.0,
            1.6038108905486153e-28,
            True
        ),
        (
            [[50, 50, 50], [50, 50, 50]],
            [0.5, 0.5, 0.5],
            0.0,
            0.0,
            1.0,
            False
        )
    ],
    ids=['significant', 'non-significant']
)
def test_run_query_global_testing_returns_valid_results(
    frequencies: list[list[int]],
    expected_rates: list[float],
    expected_association: float,
    expected_statistic: float,
    expected_p_value: float,
    expected_reject: bool,
    statistical_testing_config: StatisticalTestingConfig
) -> None:
    table = make_table(frequencies)

    result = run_statistical_testing.run_query_global_testing(
        query_contingency_table=table,
        statistical_testing_config=statistical_testing_config
    )

    assert_global_result(
        result=result,
        table=table,
        expected_rates=expected_rates,
        expected_association=expected_association,
        expected_statistic=expected_statistic,
        expected_p_value=expected_p_value,
        expected_frequencies=[[50.0, 50.0, 50.0], [50.0, 50.0, 50.0]],
        expected_reject=expected_reject
    )


@pytest.mark.parametrize(
    ('frequencies', 'expected_rates'),
    [
        (
            [[0, 0, 0], [10, 20, 30]],
            [0.0, 0.0, 0.0]
        ),
        (
            [[10, 20, 30], [0, 0, 0]],
            [1.0, 1.0, 1.0]
        )
    ],
    ids=['no-overlaps', 'all-overlap']
)
def test_run_query_global_testing_skips_degenerate_tables(
    frequencies: list[list[int]],
    expected_rates: list[float],
    statistical_testing_config: StatisticalTestingConfig
) -> None:
    table = make_table(frequencies)

    result = run_statistical_testing.run_query_global_testing(
        query_contingency_table=table,
        statistical_testing_config=statistical_testing_config
    )

    assert result['reference_names'] == table.reference_names
    assert result['overlap_statuses'] == table.overlap_statuses
    np.testing.assert_array_equal(
        result['observed_frequencies'],
        table.observed_frequencies
    )
    np.testing.assert_array_equal(
        result['overlap_rates'],
        np.array(expected_rates)
    )
    assert result['association_test_result'] is None
    assert result['chi2_test_result'] is None
    assert result['reject_null'] is None


@pytest.mark.parametrize(
    ('config_change', 'expected_association_statistic'),
    [
        ({'association_statistic': 'pearson'}, 'pearson'),
        ({'power_divergence_lambda': 2 / 3}, 'cramer'),
        ({'resampling_method': PermutationMethod()}, 'cramer')
    ],
    ids=[
        'pearson',
        'power-divergence',
        'permutation'
    ]
)
def test_run_query_global_testing_forwards_test_configuration(
    config_change: dict[str, object],
    expected_association_statistic: AssociationStatistic,
    statistical_testing_config: StatisticalTestingConfig,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    table = make_table([[8, 4, 6], [2, 6, 4]])
    config = replace(statistical_testing_config, **config_change)
    expected_frequencies = np.array([[6.0, 6.0, 6.0], [4.0, 4.0, 4.0]])
    chi2_result = Chi2ContingencyResult(
        np.float64(1.25),
        np.float64(0.5),
        2,
        expected_frequencies
    )
    association_test = MagicMock(return_value=np.float64(0.25))
    chi2_test = MagicMock(return_value=chi2_result)
    monkeypatch.setattr(
        run_statistical_testing,
        'association',
        association_test
    )
    monkeypatch.setattr(
        run_statistical_testing,
        'chi2_contingency',
        chi2_test
    )

    result = run_statistical_testing.run_query_global_testing(
        query_contingency_table=table,
        statistical_testing_config=config
    )

    association_test.assert_called_once()
    association_call = association_test.call_args
    assert association_call.args[0] is table.observed_frequencies
    assert association_call.kwargs == {
        'method': expected_association_statistic,
        'correction': False,
        'lambda_': config.power_divergence_lambda
    }

    chi2_test.assert_called_once()
    chi2_call = chi2_test.call_args
    assert chi2_call.args[0] is table.observed_frequencies
    assert chi2_call.kwargs['correction'] is False
    assert (
        chi2_call.kwargs['lambda_']
        == config.power_divergence_lambda
    )
    assert chi2_call.kwargs['method'] is config.resampling_method

    assert result['association_test_result'] == 0.25
    assert result['chi2_test_result'] is chi2_result
    assert result['reject_null'] is False


@pytest.mark.parametrize(
    ('names', 'requested_yates', 'expected_yates'),
    [
        (('reference_a', 'reference_b'), True, True),
        (
            ('reference_a', 'reference_b', 'reference_c'),
            True,
            False
        ),
        (('reference_a', 'reference_b'), False, False),
        (
            ('reference_a', 'reference_b', 'reference_c'),
            False,
            False
        )
    ],
    ids=[
        'two-references-enabled',
        'three-references-forced-off',
        'two-references-disabled',
        'three-references-disabled'
    ]
)
def test_run_query_global_testing_applies_yates_only_to_two_references(
    names: tuple[str, ...],
    requested_yates: bool,
    expected_yates: bool,
    statistical_testing_config: StatisticalTestingConfig,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    frequencies = (
        [[8, 2], [2, 8]]
        if len(names) == 2
        else [[8, 4, 6], [2, 6, 4]]
    )
    table = make_table(frequencies, names)
    config = replace(
        statistical_testing_config,
        yates_correction=requested_yates
    )
    chi2_result = Chi2ContingencyResult(
        np.float64(1.25),
        np.float64(0.5),
        len(names) - 1,
        np.ones_like(table.observed_frequencies, dtype=np.float64)
    )
    association_test = MagicMock(return_value=np.float64(0.25))
    chi2_test = MagicMock(return_value=chi2_result)
    monkeypatch.setattr(
        run_statistical_testing,
        'association',
        association_test
    )
    monkeypatch.setattr(
        run_statistical_testing,
        'chi2_contingency',
        chi2_test
    )

    result = run_statistical_testing.run_query_global_testing(
        query_contingency_table=table,
        statistical_testing_config=config
    )

    association_test.assert_called_once()
    association_call = association_test.call_args
    assert association_call.args[0] is table.observed_frequencies
    assert association_call.kwargs == {
        'method': 'cramer',
        'correction': expected_yates,
        'lambda_': None
    }
    chi2_test.assert_called_once()
    chi2_call = chi2_test.call_args
    assert chi2_call.args[0] is table.observed_frequencies
    assert chi2_call.kwargs == {
        'correction': expected_yates,
        'lambda_': None,
        'method': None
    }
    assert result['association_test_result'] == 0.25
    assert result['chi2_test_result'] is chi2_result
    assert result['reject_null'] is False


def test_run_query_pairwise_testing_tests_and_adjusts_all_pairs(
    statistical_testing_config: StatisticalTestingConfig
) -> None:
    table = make_table([[90, 89, 10], [10, 11, 90]])

    results = run_statistical_testing.run_query_pairwise_testing(
        query_contingency_table=table,
        statistical_testing_config=statistical_testing_config
    )

    expected_results = (
        (
            ('reference_a', 'reference_b'),
            0.016310370902867074,
            0.05320563979781857,
            0.8175762492319703,
            [[89.5, 89.5], [10.5, 10.5]],
            0.8175762492319703,
            False
        ),
        (
            ('reference_a', 'reference_c'),
            0.8,
            128.0,
            1.1224297172982928e-29,
            [[50.0, 50.0], [50.0, 50.0]],
            3.3672891518948783e-29,
            True
        ),
        (
            ('reference_b', 'reference_c'),
            0.7900395029627469,
            124.83248324832483,
            5.53777092999323e-29,
            [[49.5, 49.5], [50.5, 50.5]],
            1.107554185998646e-28,
            True
        )
    )

    assert len(results) == len(expected_results)

    for result, expected in zip(results, expected_results):
        assert_pairwise_result(
            result=result,
            expected_reference_names=expected[0],
            expected_association=expected[1],
            expected_statistic=expected[2],
            expected_p_value=expected[3],
            expected_frequencies=expected[4],
            expected_adjusted_p_value=expected[5],
            expected_reject=expected[6]
        )


@pytest.mark.parametrize(
    ('resampling_method', 'expected_yates', 'expected_lambda'),
    [
        (None, True, 2 / 3),
        (PermutationMethod(), False, None)
    ],
    ids=['yates-and-power-divergence', 'permutation']
)
def test_run_query_pairwise_testing_forwards_test_configuration(
    resampling_method: PermutationMethod | None,
    expected_yates: bool,
    expected_lambda: float | None,
    statistical_testing_config: StatisticalTestingConfig,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    table = make_table([[8, 5, 2], [2, 5, 8]])
    config = replace(
        statistical_testing_config,
        yates_correction=expected_yates,
        power_divergence_lambda=expected_lambda,
        resampling_method=resampling_method,
        association_statistic='pearson',
        significance_level=0.1,
        adjust_method='bonferroni',
        adjust_max_iterations=3
    )
    expected_observed_frequencies = (
        np.array([[8, 5], [2, 5]]),
        np.array([[8, 2], [2, 8]]),
        np.array([[5, 2], [5, 8]])
    )
    chi2_results = (
        Chi2ContingencyResult(1.0, 0.01, 1, np.ones((2, 2))),
        Chi2ContingencyResult(2.0, 0.2, 1, np.ones((2, 2))),
        Chi2ContingencyResult(3.0, 0.03, 1, np.ones((2, 2)))
    )
    association_test = MagicMock(
        side_effect=(0.1, 0.2, 0.3)
    )
    chi2_test = MagicMock(side_effect=chi2_results)
    multiple_testing = MagicMock(
        return_value=(
            np.array([True, False, True]),
            np.array([0.03, 0.6, 0.09]),
            0.0,
            0.0
        )
    )
    monkeypatch.setattr(
        run_statistical_testing,
        'association',
        association_test
    )
    monkeypatch.setattr(
        run_statistical_testing,
        'chi2_contingency',
        chi2_test
    )
    monkeypatch.setattr(
        run_statistical_testing,
        'multipletests',
        multiple_testing
    )

    results = run_statistical_testing.run_query_pairwise_testing(
        query_contingency_table=table,
        statistical_testing_config=config
    )

    assert association_test.call_count == 3
    assert chi2_test.call_count == 3
    for index, expected_observed in enumerate(
        expected_observed_frequencies
    ):
        association_call = association_test.call_args_list[index]
        np.testing.assert_array_equal(
            association_call.args[0],
            expected_observed
        )
        assert association_call.kwargs == {
            'method': 'pearson',
            'correction': expected_yates,
            'lambda_': expected_lambda
        }

        chi2_call = chi2_test.call_args_list[index]
        np.testing.assert_array_equal(
            chi2_call.args[0],
            expected_observed
        )
        assert chi2_call.kwargs['correction'] is expected_yates
        assert chi2_call.kwargs['lambda_'] == expected_lambda
        assert chi2_call.kwargs['method'] is resampling_method

    multiple_testing.assert_called_once()
    multiple_testing_call = multiple_testing.call_args
    assert multiple_testing_call.args[0] == [0.01, 0.2, 0.03]
    assert multiple_testing_call.kwargs == {
        'alpha': 0.1,
        'method': 'bonferroni',
        'maxiter': 3
    }
    assert [
        float(result['adjusted_p_value'])
        for result in results
    ] == [0.03, 0.6, 0.09]
    assert [
        bool(result['reject_null'])
        for result in results
    ] == [True, False, True]


@pytest.mark.parametrize(
    ('adjust_method', 'adjust_max_iterations'),
    [
        ('holm-sidak', 1),
        ('fdr_tsbh', -1),
        ('fdr_tsbh', 0),
        ('fdr_tsbky', 3)
    ]
)
def test_run_query_pairwise_testing_forwards_adjustment_configuration(
    adjust_method: AdjustMethod,
    adjust_max_iterations: int,
    statistical_testing_config: StatisticalTestingConfig,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    table = make_table([[90, 89, 10], [10, 11, 90]])
    config = replace(
        statistical_testing_config,
        significance_level=0.1,
        adjust_method=adjust_method,
        adjust_max_iterations=adjust_max_iterations
    )
    multiple_testing = MagicMock(
        return_value=(
            np.array([False, True, True]),
            np.array([0.8, 0.02, 0.03]),
            0.0,
            0.0
        )
    )
    monkeypatch.setattr(
        run_statistical_testing,
        'multipletests',
        multiple_testing
    )

    results = run_statistical_testing.run_query_pairwise_testing(
        query_contingency_table=table,
        statistical_testing_config=config
    )

    multiple_testing.assert_called_once()
    multiple_testing_call = multiple_testing.call_args
    expected_p_values = [
        result['chi2_test_result'].pvalue
        for result in results
        if result['chi2_test_result'] is not None
    ]
    np.testing.assert_allclose(
        multiple_testing_call.args[0],
        expected_p_values,
        rtol=1e-12,
        atol=0.0
    )
    assert multiple_testing_call.kwargs == {
        'alpha': 0.1,
        'method': adjust_method,
        'maxiter': adjust_max_iterations
    }
    assert [
        float(result['adjusted_p_value'])
        for result in results
    ] == [0.8, 0.02, 0.03]
    assert [
        bool(result['reject_null'])
        for result in results
    ] == [False, True, True]


def test_run_query_pairwise_testing_keeps_degenerate_pairs_as_none(
    statistical_testing_config: StatisticalTestingConfig
) -> None:
    table = make_table([[0, 0, 5], [10, 20, 5]])
    config = replace(
        statistical_testing_config,
        adjust_method='bonferroni'
    )

    results = run_statistical_testing.run_query_pairwise_testing(
        query_contingency_table=table,
        statistical_testing_config=config
    )

    invalid, *valid = results
    assert invalid == {
        'reference_names': ('reference_a', 'reference_b'),
        'overlap_statuses': ('overlap', 'no_overlap'),
        'association_test_result': None,
        'chi2_test_result': None,
        'adjusted_p_value': None,
        'reject_null': None
    }

    expected_valid_results = (
        (
            ('reference_a', 'reference_c'),
            0.5773502691896258,
            6.666666666666667,
            0.009823274507519245,
            [[2.5, 2.5], [7.5, 7.5]],
            0.029469823522557736,
            True
        ),
        (
            ('reference_b', 'reference_c'),
            0.6324555320336759,
            11.999999999999998,
            0.0005320055051392503,
            [
                [3.3333333333333335, 1.6666666666666667],
                [16.666666666666668, 8.333333333333334]
            ],
            0.001596016515417751,
            True
        )
    )

    assert len(valid) == len(expected_valid_results)

    for result, expected in zip(valid, expected_valid_results):
        assert_pairwise_result(
            result=result,
            expected_reference_names=expected[0],
            expected_association=expected[1],
            expected_statistic=expected[2],
            expected_p_value=expected[3],
            expected_frequencies=expected[4],
            expected_adjusted_p_value=expected[5],
            expected_reject=expected[6]
        )


@pytest.mark.parametrize(
    ('frequencies', 'expected_invalid_results'),
    [
        ([[0, 0, 5], [10, 20, 5]], [True, False, False]),
        ([[0, 5, 0], [10, 5, 20]], [False, True, False]),
        ([[5, 0, 0], [5, 10, 20]], [False, False, True])
    ],
    ids=['first-pair', 'middle-pair', 'last-pair']
)
def test_run_query_pairwise_testing_preserves_invalid_pair_positions(
    frequencies: list[list[int]],
    expected_invalid_results: list[bool],
    statistical_testing_config: StatisticalTestingConfig
) -> None:
    table = make_table(frequencies)
    config = replace(
        statistical_testing_config,
        adjust_method='bonferroni'
    )

    results = run_statistical_testing.run_query_pairwise_testing(
        query_contingency_table=table,
        statistical_testing_config=config
    )

    assert [
        result['association_test_result'] is None
        for result in results
    ] == expected_invalid_results
    assert [
        result['chi2_test_result'] is None
        for result in results
    ] == expected_invalid_results
    assert [
        result['adjusted_p_value'] is None
        for result in results
    ] == expected_invalid_results
    assert [
        result['reject_null'] is None
        for result in results
    ] == expected_invalid_results


def test_run_query_pairwise_testing_handles_no_valid_pairs(
    statistical_testing_config: StatisticalTestingConfig
) -> None:
    table = make_table([[0, 0, 0], [10, 20, 30]])
    config = replace(
        statistical_testing_config,
        adjust_method='bonferroni'
    )

    results = run_statistical_testing.run_query_pairwise_testing(
        query_contingency_table=table,
        statistical_testing_config=config
    )

    assert results == [
        {
            'reference_names': ('reference_a', 'reference_b'),
            'overlap_statuses': ('overlap', 'no_overlap'),
            'association_test_result': None,
            'chi2_test_result': None,
            'adjusted_p_value': None,
            'reject_null': None
        },
        {
            'reference_names': ('reference_a', 'reference_c'),
            'overlap_statuses': ('overlap', 'no_overlap'),
            'association_test_result': None,
            'chi2_test_result': None,
            'adjusted_p_value': None,
            'reject_null': None
        },
        {
            'reference_names': ('reference_b', 'reference_c'),
            'overlap_statuses': ('overlap', 'no_overlap'),
            'association_test_result': None,
            'chi2_test_result': None,
            'adjusted_p_value': None,
            'reject_null': None
        }
    ]


def test_run_query_statistical_testing_runs_requested_pairwise_tests(
    statistical_testing_config: StatisticalTestingConfig,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    table = make_table([[90, 10, 50], [10, 90, 50]])
    global_result = run_statistical_testing.run_query_global_testing(
        query_contingency_table=table,
        statistical_testing_config=statistical_testing_config
    )
    pairwise_results = run_statistical_testing.run_query_pairwise_testing(
        query_contingency_table=table,
        statistical_testing_config=statistical_testing_config
    )
    global_testing = MagicMock(return_value=global_result)
    pairwise = MagicMock(return_value=pairwise_results)
    monkeypatch.setattr(
        run_statistical_testing,
        'run_query_global_testing',
        global_testing
    )
    monkeypatch.setattr(
        run_statistical_testing,
        'run_query_pairwise_testing',
        pairwise
    )

    result = run_statistical_testing.run_query_statistical_testing(
        query_contingency_table=table,
        statistical_testing_config=statistical_testing_config
    )

    global_testing.assert_called_once_with(
        query_contingency_table=table,
        statistical_testing_config=statistical_testing_config
    )
    pairwise.assert_called_once_with(
        query_contingency_table=table,
        statistical_testing_config=statistical_testing_config
    )
    assert result['raw_global_testing_result'] is global_result
    assert_global_result(
        result=global_result,
        table=table,
        expected_rates=[0.9, 0.1, 0.5],
        expected_association=0.6531972647421809,
        expected_statistic=128.0,
        expected_p_value=1.6038108905486153e-28,
        expected_frequencies=[[50.0, 50.0, 50.0], [50.0, 50.0, 50.0]],
        expected_reject=True
    )
    assert result['raw_pairwise_testing_results'] is pairwise_results
    assert_standard_significant_pairwise_results(pairwise_results)


@pytest.mark.parametrize(
    ('table', 'config_change', 'expected_global_reject'),
    [
        (
            make_table([[50, 50, 50], [50, 50, 50]]),
            {},
            False
        ),
        (
            make_table([[90, 10, 50], [10, 90, 50]]),
            {'pairwise_testing': False},
            True
        ),
        (
            make_table([[0, 0, 0], [10, 20, 30]]),
            {},
            None
        )
    ],
    ids=['non-significant', 'pairwise-disabled', 'global-test-invalid']
)
def test_run_query_statistical_testing_skips_unneeded_pairwise_tests(
    table: ContingencyTable,
    config_change: dict[str, bool],
    expected_global_reject: bool | None,
    statistical_testing_config: StatisticalTestingConfig,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(statistical_testing_config, **config_change)
    global_result = run_statistical_testing.run_query_global_testing(
        query_contingency_table=table,
        statistical_testing_config=config
    )
    global_testing = MagicMock(return_value=global_result)
    pairwise = MagicMock()
    monkeypatch.setattr(
        run_statistical_testing,
        'run_query_global_testing',
        global_testing
    )
    monkeypatch.setattr(
        run_statistical_testing,
        'run_query_pairwise_testing',
        pairwise
    )

    result = run_statistical_testing.run_query_statistical_testing(
        query_contingency_table=table,
        statistical_testing_config=config
    )

    global_testing.assert_called_once_with(
        query_contingency_table=table,
        statistical_testing_config=config
    )
    pairwise.assert_not_called()
    assert result['raw_global_testing_result'] is global_result
    assert (
        result['raw_global_testing_result']['reject_null']
        is expected_global_reject
    )
    assert result['raw_pairwise_testing_results'] is None


def test_run_statistical_testing_preserves_query_and_distance_names(
    statistical_testing_config: StatisticalTestingConfig
) -> None:
    significant_table = make_table(
        [[90, 10, 50], [10, 90, 50]]
    )
    non_significant_table = make_table(
        [[50, 50, 50], [50, 50, 50]]
    )
    contingency_tables = {
        'query_one': {
            '0bp': significant_table,
            '10bp': non_significant_table
        },
        'query_two': {'0bp': non_significant_table}
    }

    result = run_statistical_testing.run_statistical_testing(
        contingency_tables=contingency_tables,
        statistical_testing_config=statistical_testing_config
    )

    assert tuple(result) == ('query_one', 'query_two')
    assert tuple(result['query_one']) == ('0bp', '10bp')
    assert tuple(result['query_two']) == ('0bp',)

    significant_result = result['query_one']['0bp']
    assert_global_result(
        result=significant_result['raw_global_testing_result'],
        table=significant_table,
        expected_rates=[0.9, 0.1, 0.5],
        expected_association=0.6531972647421809,
        expected_statistic=128.0,
        expected_p_value=1.6038108905486153e-28,
        expected_frequencies=[[50.0, 50.0, 50.0], [50.0, 50.0, 50.0]],
        expected_reject=True
    )
    pairwise_results = significant_result['raw_pairwise_testing_results']
    assert pairwise_results is not None
    assert_standard_significant_pairwise_results(pairwise_results)

    for query_name, overlap_name in (
        ('query_one', '10bp'),
        ('query_two', '0bp')
    ):
        non_significant_result = result[query_name][overlap_name]
        assert_global_result(
            result=non_significant_result['raw_global_testing_result'],
            table=non_significant_table,
            expected_rates=[0.5, 0.5, 0.5],
            expected_association=0.0,
            expected_statistic=0.0,
            expected_p_value=1.0,
            expected_frequencies=[
                [50.0, 50.0, 50.0],
                [50.0, 50.0, 50.0]
            ],
            expected_reject=False
        )
        assert non_significant_result['raw_pairwise_testing_results'] is None


def test_run_statistical_testing_delegates_every_query_distance_pair(
    statistical_testing_config: StatisticalTestingConfig,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    table_a = make_table([[2, 1], [1, 2]], ('a', 'b'))
    table_b = make_table([[3, 1], [1, 3]], ('a', 'b'))
    contingency_tables = {
        'query_one': {'0bp': table_a, '10bp': table_b},
        'query_two': {'0bp': table_b}
    }
    raw_results = (
        make_raw_result(table_a),
        make_raw_result(table_b),
        make_raw_result(table_b)
    )
    calls = []

    def fake_run(
        query_contingency_table: ContingencyTable,
        statistical_testing_config: StatisticalTestingConfig
    ) -> RawStatisticalResult:
        calls.append(
            (query_contingency_table, statistical_testing_config)
        )
        return raw_results[len(calls) - 1]

    monkeypatch.setattr(
        run_statistical_testing,
        'run_query_statistical_testing',
        fake_run
    )

    result = run_statistical_testing.run_statistical_testing(
        contingency_tables=contingency_tables,
        statistical_testing_config=statistical_testing_config
    )

    assert [call[0] for call in calls] == [table_a, table_b, table_b]
    assert all(call[1] is statistical_testing_config for call in calls)
    assert result['query_one']['0bp'] is raw_results[0]
    assert result['query_one']['10bp'] is raw_results[1]
    assert result['query_two']['0bp'] is raw_results[2]
