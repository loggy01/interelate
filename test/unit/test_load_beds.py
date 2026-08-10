import logging
from pathlib import Path
from unittest.mock import MagicMock, call

import pyranges1 as pr
import pytest

from conftest import BedFileFactory
from interelate import load_beds
from interelate.load_beds import BedDirs


def test_resolve_bed_dir_returns_an_absolute_resolved_path(
    tmp_path: Path
) -> None:
    unresolved = tmp_path / 'nested' / '..'

    assert load_beds.resolve_bed_dir(
        'reference',
        unresolved
    ) == tmp_path.resolve()


@pytest.mark.parametrize(
    'exception',
    [OSError('cannot resolve'), RuntimeError('cannot resolve')],
    ids=['os-error', 'runtime-error']
)
def test_resolve_bed_dir_reports_and_propagates_resolution_errors(
    exception: OSError | RuntimeError,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_resolution(self: Path, strict: bool = False) -> Path:
        raise exception

    monkeypatch.setattr(Path, 'resolve', fail_resolution)

    with pytest.raises(type(exception)) as error:
        load_beds.resolve_bed_dir('query', Path('unresolvable'))

    assert str(error.value) == 'cannot resolve'
    assert capsys.readouterr().out == (
        'Failed while resolving --query_dir: cannot resolve\n'
    )


def test_collect_paths_accepts_supported_extensions_and_warns_about_others(
    tmp_path: Path,
    bed_file_factory: BedFileFactory,
    caplog: pytest.LogCaptureFixture
) -> None:
    one = bed_file_factory(
        tmp_path / 'one.bed',
        (('chr1', 0, 1),)
    )
    two = bed_file_factory(
        tmp_path / 'two.BED.GZ',
        (('chr1', 0, 1),)
    )
    three = bed_file_factory(
        tmp_path / 'three.txt',
        (('chr1', 0, 1),)
    )
    four = bed_file_factory(
        tmp_path / 'four.TXT.GZ',
        (('chr1', 0, 1),)
    )
    (tmp_path / 'nested.bed').mkdir()
    (tmp_path / 'notes.csv').write_text('ignored', encoding='utf-8')

    with caplog.at_level(logging.WARNING):
        paths = load_beds.collect_paths('reference', tmp_path)

    assert paths == (four, one, three, two)
    assert caplog.messages == [
        'Skipped files with unsupported extensions in '
        '--reference_dir: notes.csv'
    ]


def test_collect_paths_reports_and_propagates_directory_errors(
    capsys: pytest.CaptureFixture[str]
) -> None:
    broken_directory = MagicMock(spec=Path)
    broken_directory.iterdir.side_effect = OSError('cannot list')

    with pytest.raises(OSError) as error:
        load_beds.collect_paths('reference', broken_directory)

    assert str(error.value) == 'cannot list'
    assert capsys.readouterr().out == (
        'Failed while collecting file paths from '
        '--reference_dir: cannot list\n'
    )


@pytest.mark.parametrize(
    ('filename', 'expected'),
    [
        ('sample.bed', 'sample'),
        (' sample name.BED.GZ ', 'sample_name'),
        ('query.v1.txt', 'query.v1'),
        ('odd @ name!.txt.gz', 'odd__name'),
        ('already-clean', 'already-clean')
    ]
)
def test_sanitise_filename(filename: str, expected: str) -> None:
    assert load_beds.sanitise_filename(Path(filename)) == expected


def test_sanitise_filename_rejects_names_without_valid_characters() -> None:
    path = Path('@@@.bed')

    with pytest.raises(ValueError) as error:
        load_beds.sanitise_filename(path)

    assert str(error.value) == (
        f'Filename from {path} does not contain at least one valid '
        'character (letters, numbers, underscore, dots, and dashes), '
        'excluding extensions.'
    )


def test_check_filename_unique_accepts_a_new_name(tmp_path: Path) -> None:
    result = load_beds.check_filename_unique(
        path=tmp_path / 'new.bed',
        filename='new',
        seen_names={'existing': tmp_path / 'existing.bed'}
    )

    assert result is None


def test_check_filename_unique_rejects_a_duplicate_name(
    tmp_path: Path
) -> None:
    original = tmp_path / 'same.bed'
    duplicate = tmp_path / 'same.txt'
    filename = 'same'

    with pytest.raises(ValueError) as error:
        load_beds.check_filename_unique(
            path=duplicate,
            filename=filename,
            seen_names={filename: original}
        )

    assert str(error.value) == (
        f'Filenames from {duplicate} and {original} both sanitise to '
        f'"{filename}". Each filename must be unique after sanitisation '
        '(replacement of white space with underscores, removal of '
        'characters other than letters, numbers, underscore, dots, and '
        'dashes), and removal of extensions.'
    )


@pytest.mark.parametrize(
    'filename',
    ['valid.bed', 'valid.txt', 'valid.bed.gz', 'valid.txt.gz']
)
def test_read_valid_bed_loads_supported_files_with_integer_coordinates(
    filename: str,
    tmp_path: Path,
    bed_file_factory: BedFileFactory
) -> None:
    path = bed_file_factory(
        tmp_path / filename,
        (('chr1', 0, 10), ('chr2', 20, 30))
    )

    bed = load_beds.read_valid_bed(path)

    assert bed.shape == (2, 3)
    assert bed['Chromosome'].tolist() == ['chr1', 'chr2']
    assert bed['Start'].tolist() == [0, 20]
    assert bed['End'].tolist() == [10, 30]


def test_read_valid_bed_propagates_reader_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / 'bad.bed'

    def fail_read(_path: Path) -> pr.PyRanges:
        raise ValueError('bad BED')

    monkeypatch.setattr(load_beds.pr, 'read_bed', fail_read)

    with pytest.raises(ValueError) as error:
        load_beds.read_valid_bed(path)

    assert str(error.value) == 'bad BED'
    assert capsys.readouterr().out == (
        f'Failed while reading file {path}: bad BED\n'
    )


def test_read_valid_bed_rejects_pyranges_validation_issues(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    path = Path('invalid.bed')
    issue = 'Start is greater than End'
    bed = MagicMock(spec=pr.PyRanges)
    bed.reasons_why_frame_is_invalid.return_value = issue
    monkeypatch.setattr(
        load_beds.pr,
        'read_bed',
        lambda _path: bed
    )

    with pytest.raises(ValueError) as error:
        load_beds.read_valid_bed(path)

    assert str(error.value) == f'BED file {path} is invalid: {issue}'


def test_read_valid_bed_rejects_non_integer_coordinates(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    bed = MagicMock(spec=pr.PyRanges)
    bed.reasons_why_frame_is_invalid.return_value = None
    selected_columns = bed.iloc.__getitem__.return_value
    selected_columns.dtypes.eq.return_value.all.return_value = False
    monkeypatch.setattr(
        load_beds.pr,
        'read_bed',
        lambda _path: bed
    )

    with pytest.raises(TypeError) as error:
        load_beds.read_valid_bed(Path('floating-point.bed'))

    assert str(error.value) == (
        'Columns 2 and/or 3 are not integers in standard decimal format.'
    )


def test_load_beds_loads_and_sanitises_all_files(
    bed_directories: BedDirs,
    caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        beds = load_beds.load_beds(bed_directories)

    assert tuple(beds['reference']) == (
        'reference_one',
        'reference-two',
        'reference_three'
    )
    assert tuple(beds['query']) == ('query',)
    assert {
        reference_name: (
            bed['Chromosome'].tolist(),
            bed['Start'].tolist(),
            bed['End'].tolist()
        )
        for reference_name, bed in beds['reference'].items()
    } == {
        'reference_one': (
            ['chr1', 'chr1', 'chr1'],
            [0, 20, 40],
            [10, 30, 50]
        ),
        'reference-two': (
            ['chr1', 'chr1', 'chr1'],
            [5, 60, 80],
            [15, 70, 90]
        ),
        'reference_three': (
            ['chr1', 'chr1', 'chr1'],
            [8, 41, 100],
            [12, 55, 110]
        )
    }
    query = beds['query']['query']
    assert query['Chromosome'].tolist() == ['chr1', 'chr1']
    assert query['Start'].tolist() == [8, 42]
    assert query['End'].tolist() == [9, 43]

    expected_messages = [
        f'Loaded {bed_type} BED file from {path.resolve()} as '
        f'"{load_beds.sanitise_filename(path)}".'
        for bed_type, directory in bed_directories.items()
        for path in sorted(
            directory.iterdir(),
            key=lambda item: item.name.casefold()
        )
        if path.is_file()
    ]
    loaded_messages = [
        message
        for message in caplog.messages
        if message.startswith('Loaded ')
    ]
    assert loaded_messages == expected_messages


def test_load_beds_delegates_every_bed_type_file_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture
) -> None:
    reference_dir = tmp_path / 'references'
    query_dir = tmp_path / 'queries'
    resolved_reference_dir = tmp_path / 'resolved-references'
    resolved_query_dir = tmp_path / 'resolved-queries'
    reference_paths = (
        resolved_reference_dir / 'reference_a.bed',
        resolved_reference_dir / 'reference_b.bed'
    )
    query_paths = (resolved_query_dir / 'query.bed',)
    reference_a = pr.PyRanges(
        {'Chromosome': ['chr1'], 'Start': [0], 'End': [10]}
    )
    reference_b = pr.PyRanges(
        {'Chromosome': ['chr1'], 'Start': [20], 'End': [30]}
    )
    query = pr.PyRanges(
        {'Chromosome': ['chr1'], 'Start': [5], 'End': [6]}
    )
    resolve = MagicMock(
        side_effect=(resolved_reference_dir, resolved_query_dir)
    )
    collect = MagicMock(side_effect=(reference_paths, query_paths))
    sanitise = MagicMock(
        side_effect=('reference_a', 'reference_b', 'query')
    )
    unique_calls = []

    def check_unique(
        path: Path,
        filename: str,
        seen_names: dict[str, Path]
    ) -> None:
        unique_calls.append((path, filename, dict(seen_names)))

    read = MagicMock(side_effect=(reference_a, reference_b, query))
    monkeypatch.setattr(load_beds, 'resolve_bed_dir', resolve)
    monkeypatch.setattr(load_beds, 'collect_paths', collect)
    monkeypatch.setattr(load_beds, 'sanitise_filename', sanitise)
    monkeypatch.setattr(load_beds, 'check_filename_unique', check_unique)
    monkeypatch.setattr(load_beds, 'read_valid_bed', read)

    with caplog.at_level(logging.INFO):
        beds = load_beds.load_beds(
            {'reference': reference_dir, 'query': query_dir}
        )

    assert resolve.call_args_list == [
        call(bed_type='reference', bed_dir=reference_dir),
        call(bed_type='query', bed_dir=query_dir)
    ]
    assert collect.call_args_list == [
        call('reference', reference_dir),
        call('query', query_dir)
    ]
    assert sanitise.call_args_list == [
        call(reference_paths[0]),
        call(reference_paths[1]),
        call(query_paths[0])
    ]
    assert unique_calls == [
        (reference_paths[0], 'reference_a', {}),
        (
            reference_paths[1],
            'reference_b',
            {'reference_a': reference_paths[0]}
        ),
        (
            query_paths[0],
            'query',
            {
                'reference_a': reference_paths[0],
                'reference_b': reference_paths[1]
            }
        )
    ]
    assert read.call_args_list == [
        call(reference_paths[0]),
        call(reference_paths[1]),
        call(query_paths[0])
    ]
    assert beds['reference']['reference_a'] is reference_a
    assert beds['reference']['reference_b'] is reference_b
    assert beds['query']['query'] is query
    assert caplog.messages == [
        f'Loaded reference BED file from {reference_paths[0]} as '
        '"reference_a".',
        f'Loaded reference BED file from {reference_paths[1]} as '
        '"reference_b".',
        f'Loaded query BED file from {query_paths[0]} as "query".'
    ]


def test_load_beds_rejects_the_same_directory_for_both_types(
    tmp_path: Path
) -> None:
    with pytest.raises(ValueError) as error:
        load_beds.load_beds(
            {'reference': tmp_path, 'query': tmp_path}
        )

    assert str(error.value) == (
        '--reference_dir and --query_dir resolve to the same path.'
    )


def test_load_beds_requires_at_least_two_references(
    tmp_path: Path,
    bed_file_factory: BedFileFactory
) -> None:
    reference_dir = tmp_path / 'references'
    query_dir = tmp_path / 'queries'
    bed_file_factory(
        reference_dir / 'only.bed',
        (('chr1', 0, 10),)
    )
    bed_file_factory(
        query_dir / 'query.bed',
        (('chr1', 0, 10),)
    )

    with pytest.raises(ValueError) as error:
        load_beds.load_beds(
            {'reference': reference_dir, 'query': query_dir}
        )

    assert str(error.value) == (
        'Received fewer than two reference BED files.'
    )


def test_load_beds_requires_at_least_one_query(
    tmp_path: Path,
    bed_file_factory: BedFileFactory
) -> None:
    reference_dir = tmp_path / 'references'
    query_dir = tmp_path / 'queries'
    query_dir.mkdir()
    bed_file_factory(
        reference_dir / 'one.bed',
        (('chr1', 0, 10),)
    )
    bed_file_factory(
        reference_dir / 'two.bed',
        (('chr1', 20, 30),)
    )

    with pytest.raises(ValueError) as error:
        load_beds.load_beds(
            {'reference': reference_dir, 'query': query_dir}
        )

    assert str(error.value) == 'Received fewer than one query BED file.'


def test_load_beds_requires_names_to_be_unique_across_bed_types(
    tmp_path: Path,
    bed_file_factory: BedFileFactory
) -> None:
    reference_dir = tmp_path / 'references'
    query_dir = tmp_path / 'queries'
    original = reference_dir / 'same.bed'
    duplicate = query_dir / 'same.txt'
    filename = 'same'

    bed_file_factory(original, (('chr1', 0, 10),))
    bed_file_factory(
        reference_dir / 'other.bed',
        (('chr1', 20, 30),)
    )
    bed_file_factory(duplicate, (('chr1', 0, 10),))

    with pytest.raises(ValueError) as error:
        load_beds.load_beds(
            {'reference': reference_dir, 'query': query_dir}
        )

    assert str(error.value) == (
        f'Filenames from {duplicate} and {original} both sanitise to '
        f'"{filename}". Each filename must be unique after sanitisation '
        '(replacement of white space with underscores, removal of '
        'characters other than letters, numbers, underscore, dots, and '
        'dashes), and removal of extensions.'
    )
