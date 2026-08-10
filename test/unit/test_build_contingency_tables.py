import numpy as np
import pyranges1 as pr
import pytest

from interelate import build_contingency_tables
from interelate.build_contingency_tables import ContingencyTable


def test_build_query_contingency_table_counts_overlap_statuses() -> None:
    query_overlap_counts = {
        'reference_a': pr.PyRanges(
            {
                'Chromosome': ['chr1', 'chr1', 'chr1'],
                'Start': [0, 20, 40],
                'End': [10, 30, 50],
                '0bp': [1, 0, 2]
            }
        ),
        'reference_b': pr.PyRanges(
            {
                'Chromosome': ['chr1', 'chr1', 'chr1'],
                'Start': [5, 60, 80],
                'End': [15, 70, 90],
                '0bp': [0, 0, 1]
            }
        )
    }

    result = build_contingency_tables.build_query_contingency_table(
        query_overlap_counts=query_overlap_counts,
        overlap_name='0bp'
    )

    assert result.reference_names == ('reference_a', 'reference_b')
    assert result.overlap_statuses == ('overlap', 'no_overlap')
    np.testing.assert_array_equal(
        result.observed_frequencies,
        np.array([[2, 1], [1, 2]])
    )


def test_build_contingency_tables_preserves_query_and_overlap_names() -> None:
    overlap_counts = {
        'query_one': {
            'reference_a': pr.PyRanges(
                {
                    'Chromosome': ['chr1', 'chr1', 'chr1'],
                    'Start': [0, 20, 40],
                    'End': [10, 30, 50],
                    '0bp': [1, 0, 2],
                    '100bp': [1, 1, 2]
                }
            ),
            'reference_b': pr.PyRanges(
                {
                    'Chromosome': ['chr1', 'chr1', 'chr1'],
                    'Start': [5, 60, 80],
                    'End': [15, 70, 90],
                    '0bp': [0, 0, 1],
                    '100bp': [1, 1, 1]
                }
            )
        },
        'query_two': {
            'reference_a': pr.PyRanges(
                {
                    'Chromosome': ['chr1', 'chr1', 'chr1'],
                    'Start': [0, 20, 40],
                    'End': [10, 30, 50],
                    '0bp': [0, 0, 0],
                    '100bp': [1, 1, 1]
                }
            ),
            'reference_b': pr.PyRanges(
                {
                    'Chromosome': ['chr1', 'chr1', 'chr1'],
                    'Start': [5, 60, 80],
                    'End': [15, 70, 90],
                    '0bp': [1, 1, 0],
                    '100bp': [2, 0, 0]
                }
            )
        }
    }

    result = build_contingency_tables.build_contingency_tables(
        overlap_counts=overlap_counts,
        genomic_distances=(0, 100)
    )

    assert tuple(result) == ('query_one', 'query_two')
    assert tuple(result['query_one']) == ('0bp', '100bp')
    assert tuple(result['query_two']) == ('0bp', '100bp')
    for query_results in result.values():
        for table in query_results.values():
            assert table.reference_names == ('reference_a', 'reference_b')
            assert table.overlap_statuses == ('overlap', 'no_overlap')

    np.testing.assert_array_equal(
        result['query_one']['0bp'].observed_frequencies,
        np.array([[2, 1], [1, 2]])
    )
    np.testing.assert_array_equal(
        result['query_one']['100bp'].observed_frequencies,
        np.array([[3, 3], [0, 0]])
    )
    np.testing.assert_array_equal(
        result['query_two']['0bp'].observed_frequencies,
        np.array([[0, 2], [3, 1]])
    )
    np.testing.assert_array_equal(
        result['query_two']['100bp'].observed_frequencies,
        np.array([[3, 1], [0, 2]])
    )


def test_build_contingency_tables_delegates_every_query_distance_pair(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    counts_a = pr.PyRanges(
        {
            'Chromosome': ['chr1'],
            'Start': [0],
            'End': [10],
            '0bp': [1],
            '100bp': [1]
        }
    )
    counts_b = pr.PyRanges(
        {
            'Chromosome': ['chr1'],
            'Start': [20],
            'End': [30],
            '0bp': [0],
            '100bp': [1]
        }
    )
    overlap_counts = {
        'query_a': {
            'reference_a': counts_a,
            'reference_b': counts_b
        },
        'query_b': {
            'reference_a': counts_b,
            'reference_b': counts_a
        }
    }
    tables = tuple(
        ContingencyTable(
            reference_names=('reference_a', 'reference_b'),
            observed_frequencies=np.array([[index, 1], [1, index]])
        )
        for index in range(4)
    )
    calls = []

    def fake_build(
        query_overlap_counts: dict[str, pr.PyRanges],
        overlap_name: str
    ) -> ContingencyTable:
        calls.append((query_overlap_counts, overlap_name))
        return tables[len(calls) - 1]

    monkeypatch.setattr(
        build_contingency_tables,
        'build_query_contingency_table',
        fake_build
    )

    result = build_contingency_tables.build_contingency_tables(
        overlap_counts=overlap_counts,
        genomic_distances=(0, 100)
    )

    assert calls == [
        (overlap_counts['query_a'], '0bp'),
        (overlap_counts['query_a'], '100bp'),
        (overlap_counts['query_b'], '0bp'),
        (overlap_counts['query_b'], '100bp')
    ]
    assert result['query_a']['0bp'] is tables[0]
    assert result['query_a']['100bp'] is tables[1]
    assert result['query_b']['0bp'] is tables[2]
    assert result['query_b']['100bp'] is tables[3]
