import json
from pathlib import Path
from typing import Any

import pytest

from conftest import BedFileFactory
from conftest import CliConfigurator
from conftest import PipelineBedDirectoriesFactory
from interelate import cli
from interelate.load_beds import BedDirs
from interelate.run_statistical_testing import StatisticalTestingConfig
from interelate.write_output import RUN_LOG_HANDLER


pytestmark = pytest.mark.integration


REFERENCE_FILENAMES = (
    'reference_a.bed',
    'reference_b.bed',
    'reference_c.bed.gz'
)


def run_application(
    bed_directories: BedDirs,
    output_dir: Path,
    cli_configurator: CliConfigurator,
    capsys: pytest.CaptureFixture[str],
    genomic_distances: tuple[str, ...] = ('0',),
    **overrides: object
) -> str:
    RUN_LOG_HANDLER.buffer.clear()
    cli_configurator(
        bed_directories['reference'],
        bed_directories['query'],
        output_dir,
        real_logging=True,
        GENOMIC_DISTANCES=list(genomic_distances),
        **overrides
    )

    result = cli.main(None)

    assert result is None
    return capsys.readouterr().out


def read_statistical_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def comparison_values(
    values: list[list[Any]]
) -> dict[tuple[str, str], Any]:
    return {
        tuple(comparison): value
        for comparison, value in values
    }


def assert_overlap_result(
    result: dict[str, Any],
    expected_overlap: dict[str, int],
    expected_no_overlap: dict[str, int],
    expected_rates: dict[str, float]
) -> None:
    overlap_result = result['overlap_result']
    observed = overlap_result['observed_frequencies']
    assert observed['overlap'] == [
        [name, value]
        for name, value in expected_overlap.items()
    ]
    assert observed['no_overlap'] == [
        [name, value]
        for name, value in expected_no_overlap.items()
    ]
    actual_rates = overlap_result['overlap_rate']
    assert [name for name, _ in actual_rates] == list(expected_rates)
    assert [value for _, value in actual_rates] == pytest.approx(
        list(expected_rates.values())
    )


def assert_valid_global_result(
    global_result: dict[str, Any],
    expected_frequencies: dict[str, dict[str, float]],
    expected_statistic: float,
    expected_p_value: float,
    expected_reject: bool,
    expected_association: float
) -> None:
    for overlap_status in ('overlap', 'no_overlap'):
        actual_frequencies = global_result[
            'expected_frequencies'
        ][overlap_status]
        expected = expected_frequencies[overlap_status]
        assert [name for name, _ in actual_frequencies] == list(expected)
        assert [value for _, value in actual_frequencies] == pytest.approx(
            list(expected.values())
        )

    assert global_result['chi2_statistic'] == pytest.approx(
        expected_statistic
    )
    assert global_result['dof'] == 2
    assert global_result['p_value'] == pytest.approx(expected_p_value)
    assert global_result['reject_null'] is expected_reject
    assert global_result['association_statistic'] == pytest.approx(
        expected_association
    )


def assert_comparison_float_values(
    values: list[list[Any]],
    expected_values: dict[tuple[str, str], float | None]
) -> None:
    actual_values = comparison_values(values)
    assert tuple(actual_values) == tuple(expected_values)

    for comparison, expected_value in expected_values.items():
        if expected_value is None:
            assert actual_values[comparison] is None
        else:
            assert actual_values[comparison] == pytest.approx(
                expected_value
            )


def assert_pairwise_expected_frequencies(
    rows: list[list[list[Any]]],
    expected_rows: dict[tuple[str, str], dict[str, float | None]]
) -> None:
    actual_rows = {
        tuple(name for name, _ in row): dict(row)
        for row in rows
    }
    assert tuple(actual_rows) == tuple(expected_rows)

    for comparison, expected_row in expected_rows.items():
        assert tuple(actual_rows[comparison]) == tuple(expected_row)
        for name, expected_value in expected_row.items():
            if expected_value is None:
                assert actual_rows[comparison][name] is None
            else:
                assert actual_rows[comparison][name] == pytest.approx(
                    expected_value
                )


def assert_output_filenames(
    output_dir: Path,
    expected_statistical_filenames: set[str],
    expected_overlap_filenames: set[str]
) -> None:
    assert {
        path.name for path in output_dir.iterdir()
    } == expected_statistical_filenames | {
        'interelate.log',
        'overlap_counts'
    }
    assert {
        path.name
        for path in (output_dir / 'overlap_counts').iterdir()
    } == expected_overlap_filenames


def test_cli_pipeline_reports_an_invalid_global_result(
    pipeline_bed_directories_factory: PipelineBedDirectoriesFactory,
    cli_configurator: CliConfigurator,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path
) -> None:
    bed_directories = pipeline_bed_directories_factory(
        {'query.bed': ((0, 0, 0), (10, 20, 30))},
        REFERENCE_FILENAMES
    )
    output_dir = tmp_path / 'invalid-global-results'

    stdout = run_application(
        bed_directories=bed_directories,
        output_dir=output_dir,
        cli_configurator=cli_configurator,
        capsys=capsys
    )

    assert_output_filenames(
        output_dir=output_dir,
        expected_statistical_filenames={'query_0bp.json'},
        expected_overlap_filenames={
            'query_reference_a.txt',
            'query_reference_b.txt',
            'query_reference_c.txt'
        }
    )
    result = read_statistical_result(output_dir / 'query_0bp.json')
    assert_overlap_result(
        result=result,
        expected_overlap={
            'reference_a': 0,
            'reference_b': 0,
            'reference_c': 0
        },
        expected_no_overlap={
            'reference_a': 10,
            'reference_b': 20,
            'reference_c': 30
        },
        expected_rates={
            'reference_a': 0.0,
            'reference_b': 0.0,
            'reference_c': 0.0
        }
    )
    assert result['global_testing_result'] is None
    assert result['pairwise_testing_result'] is None

    reference_a_lines = (
        output_dir / 'overlap_counts' / 'query_reference_a.txt'
    ).read_text(encoding='utf-8').splitlines()
    assert reference_a_lines[0].split('\t') == [
        'Chromosome',
        'Start',
        'End',
        '0bp'
    ]
    assert [
        line.split('\t')[3]
        for line in reference_a_lines[1:]
    ] == ['0'] * 10

    config = StatisticalTestingConfig(
        significance_level=0.05,
        yates_correction=False,
        power_divergence_lambda=None,
        resampling_method=None,
        association_statistic='cramer',
        adjust_method='holm-sidak',
        pairwise_testing=True,
        adjust_max_iterations=1
    )
    reference_dir = bed_directories['reference'].resolve()
    query_dir = bed_directories['query'].resolve()
    reference_a_path = reference_dir / 'reference_a.bed'
    reference_b_path = reference_dir / 'reference_b.bed'
    reference_c_path = reference_dir / 'reference_c.bed.gz'
    query_path = query_dir / 'query.bed'
    expected_log_lines = [
        'INFO: The following genomic distances will be used after '
        'filtering for non-negative integers: (0,).',
        f'INFO: Loaded reference BED file from '
        f'{reference_a_path} as "reference_a".',
        f'INFO: Loaded reference BED file from '
        f'{reference_b_path} as "reference_b".',
        f'INFO: Loaded reference BED file from '
        f'{reference_c_path} as "reference_c".',
        f'INFO: Loaded query BED file from '
        f'{query_path} as "query".',
        'INFO: The following statistical testing configuration will be '
        f'used: {config}.',
        'INFO: Counting BED file overlaps...',
        'INFO: Building contingency tables...',
        'INFO: Running statistical tests...',
        'INFO: Formatting statistical results...',
        'INFO: Writing output files...',
        f'INFO: Done! Results written to {output_dir}'
    ]
    assert stdout.splitlines() == expected_log_lines
    assert (output_dir / 'interelate.log').read_text(
        encoding='utf-8'
    ).splitlines() == expected_log_lines


def test_cli_pipeline_reports_non_significant_queries_and_distances(
    pipeline_bed_directories_factory: PipelineBedDirectoriesFactory,
    cli_configurator: CliConfigurator,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path
) -> None:
    bed_directories = pipeline_bed_directories_factory(
        {
            'query one.bed': ((50, 50, 50), (50, 50, 50)),
            'query_two.bed': ((40, 40, 40), (60, 60, 60))
        },
        REFERENCE_FILENAMES
    )
    output_dir = tmp_path / 'non-significant-results'

    run_application(
        bed_directories=bed_directories,
        output_dir=output_dir,
        cli_configurator=cli_configurator,
        capsys=capsys,
        genomic_distances=('0', '1')
    )

    assert_output_filenames(
        output_dir=output_dir,
        expected_statistical_filenames={
            'query_one_0bp.json',
            'query_one_1bp.json',
            'query_two_0bp.json',
            'query_two_1bp.json'
        },
        expected_overlap_filenames={
            'query_one_reference_a.txt',
            'query_one_reference_b.txt',
            'query_one_reference_c.txt',
            'query_two_reference_a.txt',
            'query_two_reference_b.txt',
            'query_two_reference_c.txt'
        }
    )
    expected_frequencies = {
        'query_one': (
            {
                'reference_a': 50,
                'reference_b': 50,
                'reference_c': 50
            },
            {
                'reference_a': 50,
                'reference_b': 50,
                'reference_c': 50
            },
            {
                'reference_a': 0.5,
                'reference_b': 0.5,
                'reference_c': 0.5
            }
        ),
        'query_two': (
            {
                'reference_a': 40,
                'reference_b': 40,
                'reference_c': 40
            },
            {
                'reference_a': 60,
                'reference_b': 60,
                'reference_c': 60
            },
            {
                'reference_a': 0.4,
                'reference_b': 0.4,
                'reference_c': 0.4
            }
        )
    }

    for query_name, expected in expected_frequencies.items():
        for overlap_name in ('0bp', '1bp'):
            result = read_statistical_result(
                output_dir / f'{query_name}_{overlap_name}.json'
            )
            assert_overlap_result(
                result=result,
                expected_overlap=expected[0],
                expected_no_overlap=expected[1],
                expected_rates=expected[2]
            )
            assert result['global_testing_result'] == {
                'expected_frequencies': {
                    'overlap': [
                        [name, float(value)]
                        for name, value in expected[0].items()
                    ],
                    'no_overlap': [
                        [name, float(value)]
                        for name, value in expected[1].items()
                    ]
                },
                'chi2_statistic': 0.0,
                'dof': 2,
                'p_value': 1.0,
                'reject_null': False,
                'association_statistic': 0.0
            }
            assert result['pairwise_testing_result'] is None


def test_cli_pipeline_preserves_significance_when_pairwise_is_disabled(
    pipeline_bed_directories_factory: PipelineBedDirectoriesFactory,
    cli_configurator: CliConfigurator,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path
) -> None:
    bed_directories = pipeline_bed_directories_factory(
        {'query.bed': ((90, 10, 50), (10, 90, 50))},
        REFERENCE_FILENAMES
    )
    output_dir = tmp_path / 'pairwise-disabled-results'

    run_application(
        bed_directories=bed_directories,
        output_dir=output_dir,
        cli_configurator=cli_configurator,
        capsys=capsys,
        PAIRWISE_TESTING=False
    )

    result = read_statistical_result(output_dir / 'query_0bp.json')
    assert_overlap_result(
        result=result,
        expected_overlap={
            'reference_a': 90,
            'reference_b': 10,
            'reference_c': 50
        },
        expected_no_overlap={
            'reference_a': 10,
            'reference_b': 90,
            'reference_c': 50
        },
        expected_rates={
            'reference_a': 0.9,
            'reference_b': 0.1,
            'reference_c': 0.5
        }
    )
    assert_valid_global_result(
        global_result=result['global_testing_result'],
        expected_frequencies={
            'overlap': {
                'reference_a': 50.0,
                'reference_b': 50.0,
                'reference_c': 50.0
            },
            'no_overlap': {
                'reference_a': 50.0,
                'reference_b': 50.0,
                'reference_c': 50.0
            }
        },
        expected_statistic=128.0,
        expected_p_value=1.603810890548633e-28,
        expected_reject=True,
        expected_association=0.6531972647421809
    )
    assert result['pairwise_testing_result'] is None


def test_cli_pipeline_reports_significant_and_non_significant_pairs(
    pipeline_bed_directories_factory: PipelineBedDirectoriesFactory,
    cli_configurator: CliConfigurator,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path
) -> None:
    bed_directories = pipeline_bed_directories_factory(
        {'query.bed': ((90, 89, 10), (10, 11, 90))},
        REFERENCE_FILENAMES
    )
    output_dir = tmp_path / 'mixed-pairwise-results'

    run_application(
        bed_directories=bed_directories,
        output_dir=output_dir,
        cli_configurator=cli_configurator,
        capsys=capsys
    )

    result = read_statistical_result(output_dir / 'query_0bp.json')
    assert_valid_global_result(
        global_result=result['global_testing_result'],
        expected_frequencies={
            'overlap': {
                'reference_a': 63.0,
                'reference_b': 63.0,
                'reference_c': 63.0
            },
            'no_overlap': {
                'reference_a': 37.0,
                'reference_b': 37.0,
                'reference_c': 37.0
            }
        },
        expected_statistic=180.7807807807808,
        expected_p_value=5.545647186781064e-40,
        expected_reject=True,
        expected_association=0.7762748241458064
    )
    pairwise_result = result['pairwise_testing_result']
    assert comparison_values(pairwise_result['reject_null']) == {
        ('reference_a', 'reference_b'): False,
        ('reference_a', 'reference_c'): True,
        ('reference_b', 'reference_c'): True
    }
    assert_comparison_float_values(
        values=pairwise_result['chi2_statistic'],
        expected_values={
            ('reference_a', 'reference_b'): 0.05320563979781857,
            ('reference_a', 'reference_c'): 128.0,
            ('reference_b', 'reference_c'): 124.83248324832482
        }
    )
    assert_comparison_float_values(
        values=pairwise_result['p_value'],
        expected_values={
            ('reference_a', 'reference_b'): 0.8175762492319703,
            ('reference_a', 'reference_c'): 1.1224297172982905e-29,
            ('reference_b', 'reference_c'): 5.537770929993264e-29
        }
    )
    assert_comparison_float_values(
        values=pairwise_result['adjusted_p_value'],
        expected_values={
            ('reference_a', 'reference_b'): 0.8175762492319703,
            ('reference_a', 'reference_c'): 3.3672891518948716e-29,
            ('reference_b', 'reference_c'): 1.1075541859986528e-28
        }
    )
    assert_comparison_float_values(
        values=pairwise_result['association_statistic'],
        expected_values={
            ('reference_a', 'reference_b'): 0.016310370902867074,
            ('reference_a', 'reference_c'): 0.8,
            ('reference_b', 'reference_c'): 0.7900395029627468
        }
    )
    assert_pairwise_expected_frequencies(
        rows=pairwise_result['expected_frequencies']['overlap'],
        expected_rows={
            ('reference_a', 'reference_b'): {
                'reference_a': 89.5,
                'reference_b': 89.5
            },
            ('reference_a', 'reference_c'): {
                'reference_a': 50.0,
                'reference_c': 50.0
            },
            ('reference_b', 'reference_c'): {
                'reference_b': 49.5,
                'reference_c': 49.5
            }
        }
    )
    assert_pairwise_expected_frequencies(
        rows=pairwise_result['expected_frequencies']['no_overlap'],
        expected_rows={
            ('reference_a', 'reference_b'): {
                'reference_a': 10.5,
                'reference_b': 10.5
            },
            ('reference_a', 'reference_c'): {
                'reference_a': 50.0,
                'reference_c': 50.0
            },
            ('reference_b', 'reference_c'): {
                'reference_b': 50.5,
                'reference_c': 50.5
            }
        }
    )


def test_cli_pipeline_preserves_invalid_pairwise_comparisons(
    pipeline_bed_directories_factory: PipelineBedDirectoriesFactory,
    cli_configurator: CliConfigurator,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path
) -> None:
    bed_directories = pipeline_bed_directories_factory(
        {'query.bed': ((0, 0, 5), (2, 100, 5))},
        REFERENCE_FILENAMES
    )
    output_dir = tmp_path / 'invalid-pairwise-results'

    run_application(
        bed_directories=bed_directories,
        output_dir=output_dir,
        cli_configurator=cli_configurator,
        capsys=capsys,
        ADJUST_METHOD='bonferroni'
    )

    result = read_statistical_result(output_dir / 'query_0bp.json')
    assert_valid_global_result(
        global_result=result['global_testing_result'],
        expected_frequencies={
            'overlap': {
                'reference_a': 0.08928571428571429,
                'reference_b': 4.464285714285714,
                'reference_c': 0.44642857142857145
            },
            'no_overlap': {
                'reference_a': 1.9107142857142858,
                'reference_b': 95.53571428571429,
                'reference_c': 9.553571428571429
            }
        },
        expected_statistic=53.383177570093466,
        expected_p_value=2.5585286083316588e-12,
        expected_reject=True,
        expected_association=0.6903879445780405
    )
    pairwise_result = result['pairwise_testing_result']
    assert comparison_values(pairwise_result['reject_null']) == {
        ('reference_a', 'reference_b'): None,
        ('reference_a', 'reference_c'): False,
        ('reference_b', 'reference_c'): True
    }
    assert_comparison_float_values(
        values=pairwise_result['chi2_statistic'],
        expected_values={
            ('reference_a', 'reference_b'): None,
            ('reference_a', 'reference_c'): 1.714285714285714,
            ('reference_b', 'reference_c'): 52.3809523809524
        }
    )
    assert_comparison_float_values(
        values=pairwise_result['p_value'],
        expected_values={
            ('reference_a', 'reference_b'): None,
            ('reference_a', 'reference_c'): 0.19043026382552036,
            ('reference_b', 'reference_c'): 4.571363864550457e-13
        }
    )
    assert_comparison_float_values(
        values=pairwise_result['adjusted_p_value'],
        expected_values={
            ('reference_a', 'reference_b'): None,
            ('reference_a', 'reference_c'): 0.5712907914765611,
            ('reference_b', 'reference_c'): 1.37140915936451e-12
        }
    )
    assert_comparison_float_values(
        values=pairwise_result['association_statistic'],
        expected_values={
            ('reference_a', 'reference_b'): None,
            ('reference_a', 'reference_c'): 0.3779644730092272,
            ('reference_b', 'reference_c'): 0.6900655593423544
        }
    )
    assert_pairwise_expected_frequencies(
        rows=pairwise_result['expected_frequencies']['overlap'],
        expected_rows={
            ('reference_a', 'reference_b'): {
                'reference_a': None,
                'reference_b': None
            },
            ('reference_a', 'reference_c'): {
                'reference_a': 0.8333333333333334,
                'reference_c': 4.166666666666667
            },
            ('reference_b', 'reference_c'): {
                'reference_b': 4.545454545454546,
                'reference_c': 0.45454545454545453
            }
        }
    )
    assert_pairwise_expected_frequencies(
        rows=pairwise_result['expected_frequencies']['no_overlap'],
        expected_rows={
            ('reference_a', 'reference_b'): {
                'reference_a': None,
                'reference_b': None
            },
            ('reference_a', 'reference_c'): {
                'reference_a': 1.1666666666666667,
                'reference_c': 5.833333333333333
            },
            ('reference_b', 'reference_c'): {
                'reference_b': 95.45454545454545,
                'reference_c': 9.545454545454545
            }
        }
    )


def test_cli_pipeline_forces_pairwise_off_for_two_references(
    pipeline_bed_directories_factory: PipelineBedDirectoriesFactory,
    cli_configurator: CliConfigurator,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path
) -> None:
    bed_directories = pipeline_bed_directories_factory(
        {'query.bed': ((90, 10), (10, 90))},
        ('reference_a.bed', 'reference_b.bed')
    )
    output_dir = tmp_path / 'two-reference-results'

    stdout = run_application(
        bed_directories=bed_directories,
        output_dir=output_dir,
        cli_configurator=cli_configurator,
        capsys=capsys
    )

    result = read_statistical_result(output_dir / 'query_0bp.json')
    assert result['global_testing_result']['reject_null'] is True
    assert result['pairwise_testing_result'] is None
    expected_warning = (
        'WARNING: --pairwise_testing True forced to False as '
        '< 3 references.'
    )
    assert [
        line
        for line in stdout.splitlines()
        if line.startswith('WARNING: ')
    ] == [expected_warning]
    log_lines = (output_dir / 'interelate.log').read_text(
        encoding='utf-8'
    ).splitlines()
    assert [
        line
        for line in log_lines
        if line.startswith('WARNING: ')
    ] == [expected_warning]
    config_lines = [
        line
        for line in log_lines
        if 'statistical testing configuration' in line
    ]
    assert len(config_lines) == 1
    assert 'pairwise_testing=False' in config_lines[0]


def test_cli_pipeline_overwrites_outputs_on_rerun(
    pipeline_bed_directories_factory: PipelineBedDirectoriesFactory,
    cli_configurator: CliConfigurator,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path
) -> None:
    bed_directories = pipeline_bed_directories_factory(
        {'query.bed': ((2, 2, 2), (2, 2, 2))},
        REFERENCE_FILENAMES
    )
    output_dir = tmp_path / 'rerun-results'

    first_stdout = run_application(
        bed_directories=bed_directories,
        output_dir=output_dir,
        cli_configurator=cli_configurator,
        capsys=capsys
    )

    statistical_path = output_dir / 'query_0bp.json'
    overlap_path = (
        output_dir / 'overlap_counts' / 'query_reference_a.txt'
    )
    log_path = output_dir / 'interelate.log'
    expected_statistical_text = statistical_path.read_text(
        encoding='utf-8'
    )
    expected_overlap_text = overlap_path.read_text(encoding='utf-8')
    statistical_path.write_text('stale JSON', encoding='utf-8')
    overlap_path.write_text('stale overlap counts', encoding='utf-8')
    log_path.write_text('stale log', encoding='utf-8')

    second_stdout = run_application(
        bed_directories=bed_directories,
        output_dir=output_dir,
        cli_configurator=cli_configurator,
        capsys=capsys
    )

    assert second_stdout == first_stdout
    assert statistical_path.read_text(
        encoding='utf-8'
    ) == expected_statistical_text
    assert overlap_path.read_text(
        encoding='utf-8'
    ) == expected_overlap_text
    assert log_path.read_text(encoding='utf-8') == second_stdout


def test_cli_pipeline_propagates_a_malformed_bed_error(
    tmp_path: Path,
    bed_file_factory: BedFileFactory,
    cli_configurator: CliConfigurator
) -> None:
    reference_dir = tmp_path / 'references'
    query_dir = tmp_path / 'queries'
    output_dir = tmp_path / 'invalid-output'
    bed_file_factory(
        reference_dir / 'reference_a.bed',
        (('chr1', 0, 10),)
    )
    bed_file_factory(
        reference_dir / 'reference_b.bed',
        (('chr1', 20, 30),)
    )
    invalid_path = bed_file_factory(
        query_dir / 'query.bed',
        (('chr1', 10, 0),)
    )
    cli_configurator(
        reference_dir,
        query_dir,
        output_dir,
        real_logging=True,
        GENOMIC_DISTANCES=['0']
    )

    with pytest.raises(ValueError) as error:
        cli.main(None)

    assert str(error.value).startswith(
        f'BED file {invalid_path.resolve()} is invalid: '
    )
    assert output_dir.is_dir()
    assert tuple(output_dir.iterdir()) == ()
