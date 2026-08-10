import logging
from pathlib import Path
from typing import Literal, TypeAlias
from unittest.mock import MagicMock

import pyranges1 as pr
import pytest
from scipy.stats import MonteCarloMethod, PermutationMethod

from conftest import CliConfigurator
from interelate import cli
from interelate.load_beds import Beds
from interelate.run_statistical_testing import AdjustMethod
from interelate.run_statistical_testing import StatisticalTestingConfig


ResamplingMethodName: TypeAlias = Literal['permutation', 'monte_carlo']
ResamplingMethodClass: TypeAlias = (
    type[PermutationMethod] | type[MonteCarloMethod]
)


def make_beds(reference_count: int = 3) -> Beds:
    def make_bed(index: int) -> pr.PyRanges:
        return pr.PyRanges(
            {
                'Chromosome': ['chr1'],
                'Start': [index],
                'End': [index + 1]
            }
        )

    return {
        'reference': {
            f'reference_{index}': make_bed(index)
            for index in range(reference_count)
        },
        'query': {'query': make_bed(reference_count)}
    }


def test_main_configures_and_dispatches_pipeline(
    tmp_path: Path,
    cli_configurator: CliConfigurator,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture
) -> None:
    reference_dir = tmp_path / 'references'
    query_dir = tmp_path / 'queries'
    output_dir = tmp_path / 'output'
    cli_configurator(reference_dir, query_dir, output_dir)
    beds = make_beds()
    load = MagicMock(return_value=beds)
    pipeline = MagicMock()
    monkeypatch.setattr(cli, 'load_beds', load)
    monkeypatch.setattr(cli, 'run_pipeline', pipeline)

    with caplog.at_level(logging.INFO):
        result = cli.main(None)

    expected_config = StatisticalTestingConfig(
        significance_level=0.05,
        yates_correction=False,
        power_divergence_lambda=None,
        resampling_method=None,
        association_statistic='cramer',
        adjust_method='holm-sidak',
        pairwise_testing=True,
        adjust_max_iterations=1
    )
    assert result is None
    assert output_dir.is_dir()
    load.assert_called_once_with(
        bed_dirs={
            'reference': reference_dir,
            'query': query_dir
        }
    )
    pipeline.assert_called_once()
    call = pipeline.call_args.kwargs
    assert call['beds'] is beds
    assert call['output_dir'] == output_dir
    assert call['genomic_distances'] == (0, 25, 100)
    assert call['statistical_testing_config'] == expected_config
    assert caplog.messages == [
        'The following genomic distances will be used after filtering for '
        'non-negative integers: (0, 25, 100).',
        'The following statistical testing configuration will be used: '
        f'{expected_config}.'
    ]


def test_main_passes_a_complete_non_default_configuration(
    tmp_path: Path,
    cli_configurator: CliConfigurator,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    cli_configurator(
        tmp_path / 'references',
        tmp_path / 'queries',
        tmp_path / 'output',
        SIGNIFICANCE_LEVEL=0.1,
        YATES_CORRECTION=True,
        POWER_DIVERGENCE_LAMBDA=2 / 3,
        ASSOCIATION_STATISTIC='tschuprow',
        PAIRWISE_TESTING=False,
        ADJUST_METHOD='fdr_tsbh',
        ADJUST_MAX_ITERATIONS=-1
    )
    monkeypatch.setattr(
        cli,
        'load_beds',
        lambda **_kwargs: make_beds(2)
    )
    pipeline = MagicMock()
    monkeypatch.setattr(cli, 'run_pipeline', pipeline)

    cli.main(None)

    config = pipeline.call_args.kwargs['statistical_testing_config']
    assert config == StatisticalTestingConfig(
        significance_level=0.1,
        yates_correction=True,
        power_divergence_lambda=2 / 3,
        resampling_method=None,
        association_statistic='tschuprow',
        adjust_method='fdr_tsbh',
        pairwise_testing=False,
        adjust_max_iterations=-1
    )


@pytest.mark.parametrize(
    ('method_name', 'expected_type'),
    [
        ('permutation', PermutationMethod),
        ('monte_carlo', MonteCarloMethod)
    ]
)
def test_main_builds_requested_resampling_method(
    method_name: ResamplingMethodName,
    expected_type: ResamplingMethodClass,
    tmp_path: Path,
    cli_configurator: CliConfigurator,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    cli_configurator(
        tmp_path / 'references',
        tmp_path / 'queries',
        tmp_path / 'output',
        RESAMPLING_METHOD=method_name
    )
    monkeypatch.setattr(
        cli,
        'load_beds',
        lambda **_kwargs: make_beds()
    )
    pipeline = MagicMock()
    monkeypatch.setattr(cli, 'run_pipeline', pipeline)

    cli.main(None)

    config = pipeline.call_args.kwargs['statistical_testing_config']
    assert type(config.resampling_method) is expected_type


@pytest.mark.parametrize(
    ('conflicting_option', 'expected_message'),
    [
        (
            {'YATES_CORRECTION': True},
            '--resampling_method must be None if '
            '--yates_correction is True.'
        ),
        (
            {'POWER_DIVERGENCE_LAMBDA': 2 / 3},
            '--resampling_method must be None if '
            '--power_divergence_lambda is not None.'
        )
    ]
)
def test_main_rejects_options_incompatible_with_resampling(
    conflicting_option: dict[str, object],
    expected_message: str,
    tmp_path: Path,
    cli_configurator: CliConfigurator
) -> None:
    cli_configurator(
        tmp_path / 'references',
        tmp_path / 'queries',
        tmp_path / 'output',
        RESAMPLING_METHOD='permutation',
        **conflicting_option
    )

    with pytest.raises(ValueError) as error:
        cli.main(None)

    assert str(error.value) == expected_message


def test_main_rejects_distances_without_non_negative_integers(
    tmp_path: Path,
    cli_configurator: CliConfigurator
) -> None:
    cli_configurator(
        tmp_path / 'references',
        tmp_path / 'queries',
        tmp_path / 'output',
        GENOMIC_DISTANCES=[
            '-1',
            '1.5',
            'not-a-number',
            '',
            '1e3'
        ]
    )

    with pytest.raises(ValueError) as error:
        cli.main(None)

    assert str(error.value) == (
        '--genomic_distances contains no non-negative integers.'
    )


def test_main_propagates_output_directory_creation_errors(
    tmp_path: Path,
    cli_configurator: CliConfigurator,
    capsys: pytest.CaptureFixture[str]
) -> None:
    blocking_file = tmp_path / 'blocking-file'
    blocking_file.write_text('not a directory', encoding='utf-8')
    cli_configurator(
        tmp_path / 'references',
        tmp_path / 'queries',
        blocking_file / 'output'
    )

    with pytest.raises(OSError) as error:
        cli.main(None)

    assert capsys.readouterr().out == (
        f'Failed while creating --output_dir: {error.value}\n'
    )


def test_main_accepts_an_existing_output_directory(
    tmp_path: Path,
    cli_configurator: CliConfigurator,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    existing_path = output_dir / 'existing.txt'
    existing_path.write_text('existing', encoding='utf-8')
    cli_configurator(
        tmp_path / 'references',
        tmp_path / 'queries',
        output_dir
    )
    monkeypatch.setattr(
        cli,
        'load_beds',
        lambda **_kwargs: make_beds()
    )
    pipeline = MagicMock()
    monkeypatch.setattr(cli, 'run_pipeline', pipeline)

    cli.main(None)

    assert existing_path.read_text(encoding='utf-8') == 'existing'
    pipeline.assert_called_once()


def test_main_disables_pairwise_testing_for_only_two_references(
    tmp_path: Path,
    cli_configurator: CliConfigurator,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture
) -> None:
    cli_configurator(
        tmp_path / 'references',
        tmp_path / 'queries',
        tmp_path / 'output'
    )
    monkeypatch.setattr(
        cli,
        'load_beds',
        lambda **_kwargs: make_beds(2)
    )
    pipeline = MagicMock()
    monkeypatch.setattr(cli, 'run_pipeline', pipeline)

    with caplog.at_level(logging.WARNING):
        cli.main(None)

    config = pipeline.call_args.kwargs['statistical_testing_config']
    assert config.pairwise_testing is False
    assert caplog.messages == [
        '--pairwise_testing True forced to False as < 3 references.'
    ]


def test_main_preserves_explicitly_disabled_pairwise_testing(
    tmp_path: Path,
    cli_configurator: CliConfigurator,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture
) -> None:
    cli_configurator(
        tmp_path / 'references',
        tmp_path / 'queries',
        tmp_path / 'output',
        PAIRWISE_TESTING=False
    )
    monkeypatch.setattr(
        cli,
        'load_beds',
        lambda **_kwargs: make_beds(2)
    )
    pipeline = MagicMock()
    monkeypatch.setattr(cli, 'run_pipeline', pipeline)

    with caplog.at_level(logging.WARNING):
        cli.main(None)

    config = pipeline.call_args.kwargs['statistical_testing_config']
    assert config.pairwise_testing is False
    assert caplog.messages == []


def test_main_retains_yates_and_warns_it_is_ignored_globally(
    tmp_path: Path,
    cli_configurator: CliConfigurator,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture
) -> None:
    cli_configurator(
        tmp_path / 'references',
        tmp_path / 'queries',
        tmp_path / 'output',
        YATES_CORRECTION=True
    )
    monkeypatch.setattr(
        cli,
        'load_beds',
        lambda **_kwargs: make_beds()
    )
    pipeline = MagicMock()
    monkeypatch.setattr(cli, 'run_pipeline', pipeline)

    with caplog.at_level(logging.WARNING):
        cli.main(None)

    config = pipeline.call_args.kwargs['statistical_testing_config']
    assert config.yates_correction is True
    assert caplog.messages == [
        '--yates_correction True ignored in global testing as '
        '>= 3 references.'
    ]


def test_main_retains_yates_without_warning_for_two_references(
    tmp_path: Path,
    cli_configurator: CliConfigurator,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture
) -> None:
    cli_configurator(
        tmp_path / 'references',
        tmp_path / 'queries',
        tmp_path / 'output',
        YATES_CORRECTION=True,
        PAIRWISE_TESTING=False
    )
    monkeypatch.setattr(
        cli,
        'load_beds',
        lambda **_kwargs: make_beds(2)
    )
    pipeline = MagicMock()
    monkeypatch.setattr(cli, 'run_pipeline', pipeline)

    with caplog.at_level(logging.WARNING):
        cli.main(None)

    config = pipeline.call_args.kwargs['statistical_testing_config']
    assert config.yates_correction is True
    assert caplog.messages == []


def test_main_warns_iterations_are_ignored_for_non_two_stage_adjustment(
    tmp_path: Path,
    cli_configurator: CliConfigurator,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture
) -> None:
    holders = cli_configurator(
        tmp_path / 'references',
        tmp_path / 'queries',
        tmp_path / 'output',
        ADJUST_MAX_ITERATIONS=3
    )
    holders['ADJUST_MAX_ITERATIONS'].present = True
    monkeypatch.setattr(
        cli,
        'load_beds',
        lambda **_kwargs: make_beds()
    )
    pipeline = MagicMock()
    monkeypatch.setattr(cli, 'run_pipeline', pipeline)

    with caplog.at_level(logging.WARNING):
        cli.main(None)

    config = pipeline.call_args.kwargs['statistical_testing_config']
    assert config.adjust_method == 'holm-sidak'
    assert config.adjust_max_iterations == 3
    assert caplog.messages == [
        '--adjust_max_iterations ignored as --adjust_method is not '
        'two-stage FDR.'
    ]


@pytest.mark.parametrize(
    'adjust_method',
    ['fdr_tsbh', 'fdr_tsbky']
)
def test_main_accepts_iterations_for_two_stage_adjustment(
    adjust_method: AdjustMethod,
    tmp_path: Path,
    cli_configurator: CliConfigurator,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture
) -> None:
    holders = cli_configurator(
        tmp_path / 'references',
        tmp_path / 'queries',
        tmp_path / 'output',
        ADJUST_METHOD=adjust_method,
        ADJUST_MAX_ITERATIONS=3
    )
    holders['ADJUST_MAX_ITERATIONS'].present = True
    monkeypatch.setattr(
        cli,
        'load_beds',
        lambda **_kwargs: make_beds()
    )
    pipeline = MagicMock()
    monkeypatch.setattr(cli, 'run_pipeline', pipeline)

    with caplog.at_level(logging.WARNING):
        cli.main(None)

    config = pipeline.call_args.kwargs['statistical_testing_config']
    assert config.adjust_method == adjust_method
    assert config.adjust_max_iterations == 3
    assert caplog.messages == []


def test_cli_marks_required_flags_and_runs_main(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    mark_required = MagicMock()
    app_run = MagicMock()
    monkeypatch.setattr(
        cli.flags,
        'mark_flags_as_required',
        mark_required
    )
    monkeypatch.setattr(cli.app, 'run', app_run)

    result = cli.cli()

    assert result is None
    mark_required.assert_called_once_with(
        [
            'reference_dir',
            'query_dir',
            'output_dir',
            'genomic_distances'
        ]
    )
    app_run.assert_called_once_with(cli.main)
