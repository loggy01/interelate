import gzip
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, TypeAlias

import pyranges1 as pr
import pytest

from interelate import cli
from interelate.load_beds import BedDirs, Beds
from interelate.run_statistical_testing import StatisticalTestingConfig
from interelate.write_output import RUN_LOG_HANDLER


BedRow: TypeAlias = tuple[str, int, int]
BedFileFactory: TypeAlias = Callable[[Path, Iterable[BedRow]], Path]
FrequencyTable: TypeAlias = tuple[tuple[int, ...], tuple[int, ...]]
QueryFrequencyTables: TypeAlias = dict[str, FrequencyTable]
PipelineBedDirectoriesFactory: TypeAlias = Callable[
    [QueryFrequencyTables, tuple[str, ...]],
    BedDirs
]


class CliConfigurator(Protocol):
    def __call__(
        self,
        reference_dir: Path,
        query_dir: Path,
        output_dir: Path,
        real_logging: bool = False,
        **overrides: object
    ) -> dict[str, SimpleNamespace]:
        ...


@pytest.fixture
def cli_configurator(
    monkeypatch: pytest.MonkeyPatch
) -> CliConfigurator:
    """Return a helper that supplies parsed values to the CLI."""

    def configure_cli(
        reference_dir: Path,
        query_dir: Path,
        output_dir: Path,
        real_logging: bool = False,
        **overrides: object
    ) -> dict[str, SimpleNamespace]:
        values = {
            'REFERENCE_DIR': str(reference_dir),
            'QUERY_DIR': str(query_dir),
            'OUTPUT_DIR': str(output_dir),
            'GENOMIC_DISTANCES': [
                '100',
                '0',
                '100',
                ' 25 ',
                'invalid',
                '-1'
            ],
            'SIGNIFICANCE_LEVEL': 0.05,
            'YATES_CORRECTION': False,
            'POWER_DIVERGENCE_LAMBDA': None,
            'RESAMPLING_METHOD': None,
            'ASSOCIATION_STATISTIC': 'cramer',
            'PAIRWISE_TESTING': True,
            'ADJUST_METHOD': 'holm-sidak',
            'ADJUST_MAX_ITERATIONS': 1
        }
        values.update(overrides)

        holders = {}
        for name, value in values.items():
            holder = SimpleNamespace(value=value, present=False)
            holders[name] = holder
            monkeypatch.setattr(cli, name, holder)

        if not real_logging:
            monkeypatch.setattr(
                cli.logging,
                'basicConfig',
                lambda **_kwargs: None
            )

        return holders

    return configure_cli


@pytest.fixture
def bed_file_factory() -> BedFileFactory:
    """Return a helper that writes a three-column BED file."""

    def write_bed(path: Path, rows: Iterable[BedRow]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = ''.join(
            f'{chromosome}\t{start}\t{end}\n'
            for chromosome, start, end in rows
        )

        if path.name.lower().endswith('.gz'):
            with gzip.open(path, 'wt', encoding='utf-8') as bed_file:
                bed_file.write(text)
        else:
            path.write_text(text, encoding='utf-8')

        return path

    return write_bed


@pytest.fixture
def bed_directories(
    tmp_path: Path,
    bed_file_factory: BedFileFactory
) -> BedDirs:
    """Return valid BED directories for a small pipeline run."""

    reference_dir = tmp_path / 'references'
    query_dir = tmp_path / 'queries'

    bed_file_factory(
        reference_dir / 'reference one.bed',
        (('chr1', 0, 10), ('chr1', 20, 30), ('chr1', 40, 50))
    )
    bed_file_factory(
        reference_dir / 'reference-two.bed',
        (('chr1', 5, 15), ('chr1', 60, 70), ('chr1', 80, 90))
    )
    bed_file_factory(
        reference_dir / 'reference_three.txt.gz',
        (('chr1', 8, 12), ('chr1', 41, 55), ('chr1', 100, 110))
    )
    bed_file_factory(
        query_dir / 'query.bed',
        (('chr1', 8, 9), ('chr1', 42, 43))
    )

    return {'reference': reference_dir, 'query': query_dir}


@pytest.fixture
def pipeline_bed_directories_factory(
    tmp_path: Path,
    bed_file_factory: BedFileFactory
) -> PipelineBedDirectoriesFactory:
    """Return a helper that creates BEDs with exact frequency tables."""

    def make_bed_directories(
        query_frequency_tables: QueryFrequencyTables,
        reference_filenames: tuple[str, ...]
    ) -> BedDirs:
        if not query_frequency_tables:
            raise ValueError('At least one query frequency table is required.')

        first_table = next(iter(query_frequency_tables.values()))
        reference_count = len(reference_filenames)

        if any(len(row) != reference_count for row in first_table):
            raise ValueError(
                'Frequency-table columns must match reference filenames.'
            )

        reference_totals = tuple(
            overlap_count + no_overlap_count
            for overlap_count, no_overlap_count in zip(*first_table)
        )

        for table in query_frequency_tables.values():
            if any(len(row) != reference_count for row in table):
                raise ValueError(
                    'Frequency-table columns must match reference filenames.'
                )

            totals = tuple(
                overlap_count + no_overlap_count
                for overlap_count, no_overlap_count in zip(*table)
            )
            if totals != reference_totals:
                raise ValueError(
                    'Reference totals must match across query tables.'
                )

        reference_dir = tmp_path / 'references'
        query_dir = tmp_path / 'queries'

        for reference_index, reference_filename in enumerate(
            reference_filenames
        ):
            chromosome = f'chr{reference_index + 1}'
            rows = tuple(
                (
                    chromosome,
                    interval_index * 20,
                    interval_index * 20 + 10
                )
                for interval_index in range(
                    reference_totals[reference_index]
                )
            )
            bed_file_factory(reference_dir / reference_filename, rows)

        for query_filename, table in query_frequency_tables.items():
            rows = tuple(
                (
                    f'chr{reference_index + 1}',
                    interval_index * 20 + 2,
                    interval_index * 20 + 3
                )
                for reference_index, overlap_count in enumerate(table[0])
                for interval_index in range(overlap_count)
            )

            if not rows:
                rows = (('chr_unmatched', 0, 1),)

            bed_file_factory(query_dir / query_filename, rows)

        return {'reference': reference_dir, 'query': query_dir}

    return make_bed_directories


@pytest.fixture
def sample_beds() -> Beds:
    """Return in-memory ranges with known overlaps."""

    return {
        'reference': {
            'reference_a': pr.PyRanges(
                {
                    'Chromosome': ['chr1', 'chr1', 'chr1'],
                    'Start': [0, 20, 40],
                    'End': [10, 30, 50]
                }
            ),
            'reference_b': pr.PyRanges(
                {
                    'Chromosome': ['chr1', 'chr1', 'chr1'],
                    'Start': [5, 60, 80],
                    'End': [15, 70, 90]
                }
            )
        },
        'query': {
            'query': pr.PyRanges(
                {
                    'Chromosome': ['chr1', 'chr1'],
                    'Start': [8, 42],
                    'End': [9, 43]
                }
            )
        }
    }


@pytest.fixture
def statistical_testing_config() -> StatisticalTestingConfig:
    """Return deterministic statistical testing configuration."""

    return StatisticalTestingConfig(
        significance_level=0.05,
        yates_correction=False,
        power_divergence_lambda=None,
        resampling_method=None,
        association_statistic='cramer',
        adjust_method='holm-sidak',
        pairwise_testing=True,
        adjust_max_iterations=1
    )


@pytest.fixture(autouse=True)
def clear_run_log_handler() -> Iterator[None]:
    """Prevent buffered log messages leaking between tests."""

    RUN_LOG_HANDLER.buffer.clear()
    yield None
    RUN_LOG_HANDLER.buffer.clear()
