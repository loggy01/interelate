"""Counts overlaps between query and reference BED files at genomic distances."""

from typing import Annotated
from typing import TypeAlias

import pyranges1 as pr

from interelate.load_beds import Beds


GenomicDistances: TypeAlias = Annotated[
    tuple[int, ...],
    'Defines genomic distances for counting overlaps between reference and query '
    'BED files. It is non-empty and contains unique, non-negative, and sorted ' 
    'ascending values.'
]

OverlapCounts: TypeAlias = Annotated[
    dict[str, dict[str, pr.PyRanges]],
    'Defines mapping of overlap counts between each reference and query BED file. ' 
    'The first key is the query BED filename. The second key is the reference BED '
    'filename. The value is a PyRanges object containing the overlap count between ' 
    'between that reference and query at all genomic distances. There is at least ' 
    'two references, one query, and one genomic distance. Filenames are ' 
    'santisised and unique.'
]


def calculate_query_overlap_count(
    reference_bed: pr.PyRanges,
    query_bed: pr.PyRanges,
    genomic_distances: GenomicDistances
) -> pr.PyRanges:
    """Counts overlaps between one query BED and one reference BED."""

    query_overlap_count = reference_bed

    for genomic_distance in genomic_distances:
        query_overlap_count = query_overlap_count.count_overlaps(
            query_bed,
            slack=genomic_distance,
            overlap_col=f'{genomic_distance}bp',
            strand_behavior='ignore'
        )

    return query_overlap_count


def calculate_overlap_counts(
    beds: Beds,
    genomic_distances: GenomicDistances
) -> OverlapCounts:
    """Counts overlaps between query and reference BED files at genomic
      distances.

    Args:
        beds: Dictionary containing loaded reference and query BED files in Beds
          format.
        genomic_distances: Tuple containing genomic distances in GenomicDistances 
          format.

    Returns:
        Dictionary containing the overlap counts between each query BED file and
          each reference BED file at each genomic distance in OverlapCounts
          format.
    """

    overlap_counts = {}

    for query_name, query_bed in beds['query'].items():
        overlap_counts[query_name] = {}

        for reference_name, reference_bed in beds['reference'].items():
            overlap_counts[query_name][reference_name] = (
                calculate_query_overlap_count(
                    reference_bed=reference_bed,
                    query_bed=query_bed,
                    genomic_distances=genomic_distances
                )
            )

    return overlap_counts
