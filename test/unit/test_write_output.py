import json
import logging
from pathlib import Path
from typing import Any

import pyranges1 as pr
import pytest

from interelate import write_output
from interelate.calculate_overlap_counts import OverlapCounts
from interelate.format_statistical_results import FormattedStatisticalResult
from interelate.format_statistical_results import FormattedStatisticalResults
from interelate.load_beds import Beds


def make_formatted_result(
    reference_name: str,
    overlap_count: int,
    no_overlap_count: int
) -> FormattedStatisticalResult:
    total_count = overlap_count + no_overlap_count

    return {
        'overlap_result': {
            'observed_frequencies': {
                'overlap': [(reference_name, overlap_count)],
                'no_overlap': [(reference_name, no_overlap_count)]
            },
            'overlap_rate': [
                (reference_name, overlap_count / total_count)
            ]
        },
        'global_testing_result': None,
        'pairwise_testing_result': None
    }


def make_formatted_results() -> FormattedStatisticalResults:
    return {
        'query_one': {
            '0bp': make_formatted_result('reference_a', 1, 1),
            '100bp': make_formatted_result('reference_a', 2, 0)
        },
        'query_two': {
            '0bp': make_formatted_result('reference_b', 0, 2)
        }
    }


def test_write_overlap_counts_writes_every_query_reference_file(
    tmp_path: Path
) -> None:
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    overlap_counts = {
        'query_one': {
            'reference_a': pr.PyRanges(
                {
                    'Chromosome': ['chr1', 'chr2'],
                    'Start': [0, 20],
                    'End': [10, 30],
                    '0bp': [1, 0],
                    '25bp': [2, 1]
                }
            ),
            'reference_b': pr.PyRanges(
                {
                    'Chromosome': ['chr3'],
                    'Start': [40],
                    'End': [50],
                    '0bp': [0],
                    '25bp': [1]
                }
            )
        },
        'query_two': {
            'reference_a': pr.PyRanges(
                {
                    'Chromosome': ['chr4'],
                    'Start': [60],
                    'End': [70],
                    '0bp': [3],
                    '25bp': [4]
                }
            )
        }
    }

    write_output.write_overlap_counts(overlap_counts, output_dir)

    overlap_counts_dir = output_dir / 'overlap_counts'
    expected_contents = {
        'query_one_reference_a.txt': (
            'Chromosome\tStart\tEnd\t0bp\t25bp\n'
            'chr1\t0\t10\t1\t2\n'
            'chr2\t20\t30\t0\t1\n'
        ),
        'query_one_reference_b.txt': (
            'Chromosome\tStart\tEnd\t0bp\t25bp\n'
            'chr3\t40\t50\t0\t1\n'
        ),
        'query_two_reference_a.txt': (
            'Chromosome\tStart\tEnd\t0bp\t25bp\n'
            'chr4\t60\t70\t3\t4\n'
        )
    }

    assert overlap_counts_dir.is_dir()
    assert {
        path.name for path in overlap_counts_dir.iterdir()
    } == set(expected_contents)
    for filename, expected_text in expected_contents.items():
        assert (overlap_counts_dir / filename).read_text(
            encoding='utf-8'
        ) == expected_text


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (('reference', 1), True),
        (['reference', None], True),
        ('reference', False),
        (('reference',), False),
        (('reference', 1, 2), False),
        ((('reference_a', 'reference_b'), 1), False),
        (('reference', {'value': 1}), False)
    ]
)
def test_is_scalar_pair(value: Any, expected: bool) -> None:
    assert write_output.is_scalar_pair(value) is expected


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ((('reference_a', 'reference_b'), 0.01), True),
        ([['reference_a', 'reference_b'], None], True),
        (['reference', 1], False),
        ((('reference_a', 'reference_b'), [0.01]), False),
        ((('reference_a', 'reference_b', 'reference_c'), 0.01), False),
        (None, False)
    ]
)
def test_is_comparison_value_pair(value: Any, expected: bool) -> None:
    assert write_output.is_comparison_value_pair(value) is expected


@pytest.mark.parametrize(
    ('value', 'level', 'expected'),
    [
        ([('reference_a', 1.0), ('reference_b', 2.0)], 4, True),
        ([('reference_a', None), ('reference_b', None)], 4, True),
        ([('reference_a', 1.0), ('reference_b', 2.0)], 3, False),
        ([('reference_a', 1.0), ['nested', []]], 4, False),
        ({'reference_a': 1.0}, 4, False)
    ]
)
def test_is_expected_frequency_row(
    value: Any,
    level: int,
    expected: bool
) -> None:
    assert write_output.is_expected_frequency_row(value, level) is expected


def test_format_json_produces_exact_custom_readable_layout() -> None:
    payload = {
        'overlap_result': {
            'observed_frequencies': {
                'overlap': [('reference_a', 2), ('reference_b', 1)],
                'no_overlap': [('reference_a', 1), ('reference_b', 2)]
            },
            'overlap_rate': [
                ('reference_a', 2 / 3),
                ('reference_b', 1 / 3)
            ]
        },
        'global_testing_result': None,
        'pairwise_testing_result': {
            'p_value': [
                (('reference_a', 'reference_b'), 0.25),
                (('reference_a', 'reference_c'), None)
            ],
            'expected_frequencies': {
                'overlap': [
                    [('reference_a', 1.5), ('reference_b', 1.5)],
                    [('reference_a', None), ('reference_c', None)]
                ]
            }
        }
    }
    expected_text = '\n'.join(
        [
            '{',
            '  "overlap_result": {',
            '    "observed_frequencies": {',
            '      "overlap": [',
            '        ["reference_a", 2],',
            '        ["reference_b", 1]',
            '      ],',
            '      "no_overlap": [',
            '        ["reference_a", 1],',
            '        ["reference_b", 2]',
            '      ]',
            '    },',
            '    "overlap_rate": [',
            '      ["reference_a", 0.6666666666666666],',
            '      ["reference_b", 0.3333333333333333]',
            '    ]',
            '  },',
            '',
            '  "global_testing_result": null,',
            '',
            '  "pairwise_testing_result": {',
            '    "p_value": [',
            '      [["reference_a", "reference_b"], 0.25],',
            '      [["reference_a", "reference_c"], null]',
            '    ],',
            '    "expected_frequencies": {',
            '      "overlap": [',
            '        [["reference_a", 1.5], '
            '["reference_b", 1.5]],',
            '        [["reference_a", null], '
            '["reference_c", null]]',
            '      ]',
            '    }',
            '  }',
            '}'
        ]
    )

    text = write_output.format_json(payload)

    assert text == expected_text
    assert json.loads(text) == json.loads(json.dumps(payload))


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ({}, '{}'),
        ([], '[]'),
        ((), '[]'),
        (None, 'null'),
        (True, 'true'),
        (3, '3'),
        ('text', '"text"')
    ]
)
def test_format_json_formats_empty_and_scalar_values(
    value: Any,
    expected: str
) -> None:
    assert write_output.format_json(value) == expected


def test_write_statistical_results_writes_every_query_distance_file(
    tmp_path: Path
) -> None:
    formatted_results = make_formatted_results()

    write_output.write_statistical_results(formatted_results, tmp_path)

    expected_results = {
        'query_one_0bp.json': formatted_results['query_one']['0bp'],
        'query_one_100bp.json': formatted_results['query_one']['100bp'],
        'query_two_0bp.json': formatted_results['query_two']['0bp']
    }
    output_paths = {
        path.name: path
        for path in tmp_path.iterdir()
    }

    assert set(output_paths) == set(expected_results)
    for filename, expected_result in expected_results.items():
        text = output_paths[filename].read_text(encoding='utf-8')
        assert text == write_output.format_json(expected_result) + '\n'
        assert json.loads(text) == json.loads(json.dumps(expected_result))


def test_output_writers_overwrite_files_on_rerun(tmp_path: Path) -> None:
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    first_overlap_counts = {
        'query': {
            'reference': pr.PyRanges(
                {
                    'Chromosome': ['chr1'],
                    'Start': [0],
                    'End': [10],
                    '0bp': [1]
                }
            )
        }
    }
    second_overlap_counts = {
        'query': {
            'reference': pr.PyRanges(
                {
                    'Chromosome': ['chr2', 'chr3'],
                    'Start': [20, 40],
                    'End': [30, 50],
                    '0bp': [0, 2]
                }
            )
        }
    }
    first_formatted_results = {
        'query': {'0bp': make_formatted_result('reference', 1, 0)}
    }
    second_formatted_results = {
        'query': {'0bp': make_formatted_result('reference', 0, 2)}
    }

    write_output.write_overlap_counts(first_overlap_counts, output_dir)
    write_output.write_statistical_results(
        first_formatted_results,
        output_dir
    )
    write_output.write_overlap_counts(second_overlap_counts, output_dir)
    write_output.write_statistical_results(
        second_formatted_results,
        output_dir
    )
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(write_output.RUN_LOG_HANDLER)

    try:
        logging.info('First run')
        write_output.write_log_file(output_dir)
        write_output.RUN_LOG_HANDLER.buffer.clear()
        logging.info('Second run')
        write_output.write_log_file(output_dir)
    finally:
        root_logger.removeHandler(write_output.RUN_LOG_HANDLER)
        root_logger.setLevel(original_level)

    overlap_path = output_dir / 'overlap_counts' / 'query_reference.txt'
    assert overlap_path.read_text(encoding='utf-8') == (
        'Chromosome\tStart\tEnd\t0bp\n'
        'chr2\t20\t30\t0\n'
        'chr3\t40\t50\t2\n'
    )
    statistical_path = output_dir / 'query_0bp.json'
    assert statistical_path.read_text(encoding='utf-8') == (
        write_output.format_json(second_formatted_results['query']['0bp'])
        + '\n'
    )
    assert (output_dir / 'interelate.log').read_text(
        encoding='utf-8'
    ) == (
        'INFO: Second run\n'
        f'INFO: Done! Results written to {output_dir}\n'
    )


def test_write_log_file_writes_buffered_messages_and_completion(
    tmp_path: Path
) -> None:
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(write_output.RUN_LOG_HANDLER)

    try:
        logging.info('A buffered message')
        logging.warning('A buffered warning')
        write_output.write_log_file(tmp_path)
    finally:
        root_logger.removeHandler(write_output.RUN_LOG_HANDLER)
        root_logger.setLevel(original_level)

    assert (tmp_path / 'interelate.log').read_text(
        encoding='utf-8'
    ) == (
        'INFO: A buffered message\n'
        'WARNING: A buffered warning\n'
        f'INFO: Done! Results written to {tmp_path}\n'
    )


def test_write_output_delegates_all_output_stages(
    tmp_path: Path,
    sample_beds: Beds,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    overlap_counts = {
        'query': {'reference_a': sample_beds['reference']['reference_a']}
    }
    formatted_results = make_formatted_results()

    def fake_write_overlap_counts(
        overlap_counts: OverlapCounts,
        output_dir: Path
    ) -> None:
        calls.append(
            (
                'overlaps',
                {
                    'overlap_counts': overlap_counts,
                    'output_dir': output_dir
                }
            )
        )

    def fake_write_statistical_results(
        formatted_statistical_results: FormattedStatisticalResults,
        output_dir: Path
    ) -> None:
        calls.append(
            (
                'statistics',
                {
                    'formatted_statistical_results': (
                        formatted_statistical_results
                    ),
                    'output_dir': output_dir
                }
            )
        )

    def fake_write_log_file(output_dir: Path) -> None:
        calls.append(('log', {'output_dir': output_dir}))

    monkeypatch.setattr(
        write_output,
        'write_overlap_counts',
        fake_write_overlap_counts
    )
    monkeypatch.setattr(
        write_output,
        'write_statistical_results',
        fake_write_statistical_results
    )
    monkeypatch.setattr(
        write_output,
        'write_log_file',
        fake_write_log_file
    )

    result = write_output.write_output(
        overlap_counts,
        formatted_results,
        tmp_path
    )

    assert result is None
    assert calls == [
        (
            'overlaps',
            {
                'overlap_counts': overlap_counts,
                'output_dir': tmp_path
            }
        ),
        (
            'statistics',
            {
                'formatted_statistical_results': formatted_results,
                'output_dir': tmp_path
            }
        ),
        ('log', {'output_dir': tmp_path})
    ]
