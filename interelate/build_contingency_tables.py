"""Builds contingency tables for each query overlap counts at each genomic distance."""

from dataclasses import dataclass
from typing import Annotated
from typing import ClassVar
from typing import TypeAlias
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import pyranges1 as pr

from interelate.calculate_overlap_counts import GenomicDistances
from interelate.calculate_overlap_counts import OverlapCounts


@dataclass(frozen=True, slots=True, kw_only=True)
class ContingencyTable:
    """Stores the contingency table for one query overlap counts at one 
      genomic distance.

    Attributes:
        reference_names: Reference BED file names (≥ 2). The j-th name labels
          column j of observed_frequencies. There are at least two reference BED 
          files. Filenames are santisised and unique.
        observed_frequencies: Integer count array with shape (2, len(reference_names)).
          observed_frequencies[i, j] is the number of reference intervals with overlap status
          overlap_statuses[i] for reference_names[j].

    Class Attributes:
        overlap_statuses: Overlap status labels. The i-th status labels row i of
          observed_frequencies. Always ordered as ('overlap', 'no_overlap'), where 
          'overlap' means a reference interval overlaps at least one query interval 
          and 'no_overlap' means it overlaps no query intervals at the genomic distance 
          of interest.
    """

    reference_names: tuple[str, ...]
    observed_frequencies: NDArray[np.int_]
    overlap_statuses: ClassVar[
        tuple[Literal['overlap'], Literal['no_overlap']]
    ] = ('overlap', 'no_overlap')


ContingencyTables: TypeAlias = Annotated[
    dict[str, dict[str, ContingencyTable]],
    'Defines mapping of contingency tables for each query overlap counts at each ' 
    'genomic distance. The first key is the corresponding query BED filename. ' 
    'The second key is the genomic distance with "bp" suffix. The value is a ' 
    'ContingencyTable object containing the contingency table for that query at ' 
    'that genomic distance. There is at least one query and one genomic distance. ' 
    'Filenames are santisised and unique.'
]


def build_query_contingency_table(
    query_overlap_counts: dict[str, pr.PyRanges],
    overlap_name: str
) -> ContingencyTable:
    """Builds a contingency table for one query overlap counts at one
      genomic distance."""

    reference_names = tuple(query_overlap_counts.keys())

    observed_frequencies = np.array(
        [
            (
                np.count_nonzero(query_overlap_count[overlap_name] > 0),
                np.count_nonzero(query_overlap_count[overlap_name] == 0)
            )
            for query_overlap_count in query_overlap_counts.values()
        ],
        dtype=np.int_
    ).T

    query_contingency_table = ContingencyTable(
        reference_names=reference_names,
        observed_frequencies=observed_frequencies
    )

    return query_contingency_table


def build_contingency_tables(
    overlap_counts: OverlapCounts,
    genomic_distances: GenomicDistances
) -> ContingencyTables:
    """Builds contingency tables for each query overlap counts at each
      genomic distance.

    Args:
        overlap_counts: Dictionary containing the overlap counts between each
          query BED file and each reference BED file at each genomic distance in 
          OverlapCounts format.
        genomic_distances: Tuple containing genomic distances in GenomicDistances 
          format.  

    Returns:
        Dictionary containing contingency tables for each query overlap counts
          at each genomic distance in ContingencyTables format.
    """

    contingency_tables = {}

    for query_name, query_overlap_counts in overlap_counts.items():
        contingency_tables[query_name] = {}

        for genomic_distance in genomic_distances:
            overlap_name = f'{genomic_distance}bp'

            contingency_tables[query_name][overlap_name] = (
                build_query_contingency_table(
                    query_overlap_counts=query_overlap_counts,
                    overlap_name=overlap_name
                )
            )

    return contingency_tables
