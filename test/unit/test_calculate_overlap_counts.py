import pyranges1 as pr
import pytest

from interelate import calculate_overlap_counts
from interelate.calculate_overlap_counts import GenomicDistances
from interelate.load_beds import Beds


def test_calculate_query_overlap_count_returns_exact_distance_counts(
) -> None:
    reference = pr.PyRanges(
        {
            'Chromosome': ['chr1', 'chr1', 'chr1', 'chr2'],
            'Start': [0, 20, 40, 0],
            'End': [10, 30, 50, 10]
        }
    )
    query = pr.PyRanges(
        {
            'Chromosome': ['chr1', 'chr1', 'chr2'],
            'Start': [8, 42, 100],
            'End': [9, 43, 101]
        }
    )

    result = calculate_overlap_counts.calculate_query_overlap_count(
        reference_bed=reference,
        query_bed=query,
        genomic_distances=(0, 15)
    )

    assert result.columns.tolist() == [
        'Chromosome',
        'Start',
        'End',
        '0bp',
        '15bp'
    ]
    assert result['Chromosome'].tolist() == [
        'chr1',
        'chr1',
        'chr1',
        'chr2'
    ]
    assert result['Start'].tolist() == [0, 20, 40, 0]
    assert result['End'].tolist() == [10, 30, 50, 10]
    assert result['0bp'].tolist() == [1, 0, 1, 0]
    assert result['15bp'].tolist() == [1, 2, 1, 0]
    assert reference.columns.tolist() == ['Chromosome', 'Start', 'End']


def test_calculate_overlap_counts_preserves_all_names_and_counts(
    sample_beds: Beds
) -> None:
    sample_beds['query']['query_two'] = pr.PyRanges(
        {
            'Chromosome': ['chr1', 'chr1'],
            'Start': [65, 85],
            'End': [66, 86]
        }
    )

    result = calculate_overlap_counts.calculate_overlap_counts(
        beds=sample_beds,
        genomic_distances=(0, 25)
    )

    assert tuple(result) == ('query', 'query_two')
    assert tuple(result['query']) == ('reference_a', 'reference_b')
    assert tuple(result['query_two']) == ('reference_a', 'reference_b')
    assert result['query']['reference_a']['0bp'].tolist() == [1, 0, 1]
    assert result['query']['reference_a']['25bp'].tolist() == [1, 2, 1]
    assert result['query']['reference_b']['0bp'].tolist() == [1, 0, 0]
    assert result['query']['reference_b']['25bp'].tolist() == [1, 1, 0]
    assert result['query_two']['reference_a']['0bp'].tolist() == [0, 0, 0]
    assert result['query_two']['reference_a']['25bp'].tolist() == [0, 0, 1]
    assert result['query_two']['reference_b']['0bp'].tolist() == [0, 1, 1]
    assert result['query_two']['reference_b']['25bp'].tolist() == [0, 2, 2]


def test_calculate_overlap_counts_delegates_every_query_reference_pair(
    sample_beds: Beds,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_beds['query']['query_two'] = pr.PyRanges(
        {
            'Chromosome': ['chr1'],
            'Start': [100],
            'End': [101]
        }
    )
    calls = []
    fake_results = tuple(
        pr.PyRanges(
            {
                'Chromosome': ['chr1'],
                'Start': [index],
                'End': [index + 1]
            }
        )
        for index in range(4)
    )

    def fake_calculate(
        reference_bed: pr.PyRanges,
        query_bed: pr.PyRanges,
        genomic_distances: GenomicDistances
    ) -> pr.PyRanges:
        calls.append((reference_bed, query_bed, genomic_distances))
        return fake_results[len(calls) - 1]

    monkeypatch.setattr(
        calculate_overlap_counts,
        'calculate_query_overlap_count',
        fake_calculate
    )

    result = calculate_overlap_counts.calculate_overlap_counts(
        beds=sample_beds,
        genomic_distances=(0, 25)
    )

    expected_calls = [
        (
            sample_beds['reference']['reference_a'],
            sample_beds['query']['query'],
            (0, 25)
        ),
        (
            sample_beds['reference']['reference_b'],
            sample_beds['query']['query'],
            (0, 25)
        ),
        (
            sample_beds['reference']['reference_a'],
            sample_beds['query']['query_two'],
            (0, 25)
        ),
        (
            sample_beds['reference']['reference_b'],
            sample_beds['query']['query_two'],
            (0, 25)
        )
    ]

    assert len(calls) == len(expected_calls)
    for call, expected_call in zip(calls, expected_calls):
        assert call[0] is expected_call[0]
        assert call[1] is expected_call[1]
        assert call[2] == expected_call[2]

    assert result['query']['reference_a'] is fake_results[0]
    assert result['query']['reference_b'] is fake_results[1]
    assert result['query_two']['reference_a'] is fake_results[2]
    assert result['query_two']['reference_b'] is fake_results[3]
