import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from interelate import run_pipeline
from interelate.build_contingency_tables import ContingencyTable
from interelate.load_beds import Beds
from interelate.run_statistical_testing import StatisticalTestingConfig


def test_run_pipeline_passes_each_stage_output_to_the_next(
    tmp_path: Path,
    sample_beds: Beds,
    statistical_testing_config: StatisticalTestingConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture
) -> None:
    overlap_counts = {
        'query': {'reference_a': sample_beds['reference']['reference_a']}
    }
    table = ContingencyTable(
        reference_names=('reference_a', 'reference_b'),
        observed_frequencies=np.array([[2, 1], [1, 2]])
    )
    contingency_tables = {
        'query': {'0bp': table}
    }
    raw_results = {
        'query': {
            '0bp': {
                'raw_global_testing_result': {
                    'reference_names': table.reference_names,
                    'overlap_statuses': table.overlap_statuses,
                    'overlap_rates': np.array([2 / 3, 1 / 3]),
                    'observed_frequencies': table.observed_frequencies,
                    'association_test_result': None,
                    'chi2_test_result': None,
                    'reject_null': None
                },
                'raw_pairwise_testing_results': None
            }
        }
    }
    formatted_results = {
        'query': {
            '0bp': {
                'overlap_result': {
                    'observed_frequencies': {
                        'overlap': [
                            ('reference_a', 2),
                            ('reference_b', 1)
                        ],
                        'no_overlap': [
                            ('reference_a', 1),
                            ('reference_b', 2)
                        ]
                    },
                    'overlap_rate': [
                        ('reference_a', 2 / 3),
                        ('reference_b', 1 / 3)
                    ]
                },
                'global_testing_result': None,
                'pairwise_testing_result': None
            }
        }
    }

    stage_calls = []
    calculate = MagicMock(
        side_effect=lambda **_kwargs: (
            stage_calls.append('calculate') or overlap_counts
        )
    )
    build = MagicMock(
        side_effect=lambda **_kwargs: (
            stage_calls.append('build') or contingency_tables
        )
    )
    run_tests = MagicMock(
        side_effect=lambda **_kwargs: (
            stage_calls.append('test') or raw_results
        )
    )
    format_results = MagicMock(
        side_effect=lambda **_kwargs: (
            stage_calls.append('format') or formatted_results
        )
    )
    write = MagicMock(
        side_effect=lambda **_kwargs: stage_calls.append('write')
    )

    monkeypatch.setattr(
        run_pipeline,
        'calculate_overlap_counts',
        calculate
    )
    monkeypatch.setattr(
        run_pipeline,
        'build_contingency_tables',
        build
    )
    monkeypatch.setattr(
        run_pipeline,
        'run_statistical_testing',
        run_tests
    )
    monkeypatch.setattr(
        run_pipeline,
        'format_statistical_results',
        format_results
    )
    monkeypatch.setattr(run_pipeline, 'write_output', write)

    with caplog.at_level(logging.INFO):
        result = run_pipeline.run_pipeline(
            beds=sample_beds,
            genomic_distances=(0, 100),
            statistical_testing_config=statistical_testing_config,
            output_dir=tmp_path
        )

    assert result is None
    assert stage_calls == [
        'calculate',
        'build',
        'test',
        'format',
        'write'
    ]
    calculate.assert_called_once_with(
        beds=sample_beds,
        genomic_distances=(0, 100)
    )
    build.assert_called_once_with(
        overlap_counts=overlap_counts,
        genomic_distances=(0, 100)
    )
    run_tests.assert_called_once_with(
        contingency_tables=contingency_tables,
        statistical_testing_config=statistical_testing_config
    )
    format_results.assert_called_once_with(
        raw_statistical_results=raw_results
    )
    write.assert_called_once_with(
        overlap_counts=overlap_counts,
        formatted_statistical_results=formatted_results,
        output_dir=tmp_path
    )
    assert caplog.messages == [
        'Counting BED file overlaps...',
        'Building contingency tables...',
        'Running statistical tests...',
        'Formatting statistical results...',
        'Writing output files...'
    ]
