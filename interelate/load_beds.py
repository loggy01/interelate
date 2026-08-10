"""Loads reference and query BED files for the Interelate pipeline."""

import logging
from pathlib import Path
import re
from typing import Annotated
from typing import Literal
from typing import TypeAlias
from typing import TypedDict

import pyranges1 as pr


# Accepted file extensions to be loaded as BED files.
ACCEPTED_FILE_TYPES = ('.bed', '.bed.gz', '.txt', '.txt.gz')


BedType: TypeAlias = Annotated[
    Literal['reference', 'query'], 
    'Defines the type of BED files.'
]


class BedDirs(TypedDict):
    """Dictionary shape mapping reference and query BED types to their input
      directory paths."""
    
    reference: Path
    query: Path


class Beds(TypedDict):
    """Dictionary shape mapping reference and query BED types to
      dictionaries mapping santisised and unique BED file names to their
      loaded PyRanges objects."""
    
    reference: dict[str, pr.PyRanges]
    query: dict[str, pr.PyRanges]


def resolve_bed_dir(bed_type: BedType, bed_dir: Path) -> Path:
    """Resolves a BED type directory path."""

    try:
        bed_dir = bed_dir.resolve()
    except (OSError, RuntimeError) as e:
        print(f'Failed while resolving --{bed_type}_dir: {e}')
        raise

    return bed_dir


def collect_paths(bed_type: BedType, bed_dir: Path) -> tuple[Path, ...]:
    """Collect BED-like file paths from a BED type directory."""

    try:
        accepted_paths = []
        skipped_paths = []

        for path in bed_dir.iterdir():
            if not path.is_file():
                continue

            if path.name.lower().endswith(ACCEPTED_FILE_TYPES):
                accepted_paths.append(path)
            else:
                skipped_paths.append(path)

        if skipped_paths:
            logging.warning(
                f'Skipped files with unsupported extensions in --{bed_type}_dir: '
                + ', '.join(path.name for path in skipped_paths)
            )

        accepted_paths = tuple(
            sorted(accepted_paths, key=lambda path: path.name.casefold())
        )

        return accepted_paths

    except OSError as e:
        print(f'Failed while collecting file paths from --{bed_type}_dir: {e}')
        raise


def sanitise_filename(path: Path) -> str:
    """Sanitises a filename from a file path."""

    filename = re.sub(r'\s+', '_', path.name.strip())
    filename = re.sub(r'[^A-Za-z0-9_.-]', '', filename)
    filename = next((
        filename[: -len(suffix)]
        for suffix in ACCEPTED_FILE_TYPES 
        if filename.lower().endswith(suffix)
    ), filename)

    if filename == '':
        raise ValueError(
            f'Filename from {path} does not contain at least one valid character '
            '(letters, numbers, underscore, dots, and dashes), excluding extensions.'
        )

    return filename


def check_filename_unique(
    path: Path, 
    filename: str, 
    seen_names: dict[str, Path]
) -> None:
    """Checks that a filename is unique among seen filenames."""

    if filename in seen_names:
        raise ValueError(
            f'Filenames from {path} and {seen_names[filename]} both sanitise to '
            f'"{filename}". Each filename must be unique after sanitisation (replacement '
            'of white space with underscores, removal of characters other than '
            'letters, numbers, underscore, dots, and dashes), and removal of extensions.'
        )


def read_valid_bed(path: Path) -> pr.PyRanges:
    """Reads a PyRanges-valid BED file"""

    try:
        bed = pr.read_bed(path)
    except Exception as e:
        print(f'Failed while reading file {path}: {e}')
        raise

    bed_pr_issues = bed.reasons_why_frame_is_invalid()

    if bed_pr_issues is not None:
        raise ValueError(f'BED file {path} is invalid: {bed_pr_issues}')

    if not bed.iloc[:, [1, 2]].dtypes.eq("int64").all():
        raise TypeError("Columns 2 and/or 3 are not integers in standard decimal format.")

    return bed


def load_beds(bed_dirs: BedDirs) -> Beds:
    """Loads reference and query BED files for the Interelate pipeline.

    Args:
        bed_dirs: Dictionary containing reference and query directory paths in
          BedDirs format. Both are intended to contain top-level files to be
          loaded as BED files in the format of PyRanges objects. Subdirectories 
          and file types other than .bed, .bed.gz, .txt, and .txt.gz are ignored.

    Returns:
        Dictionary containing loaded reference and query BED files in 
          Beds format.

    Raises:
        Exception: If any exception is raised when reading a file as a PyRanges 
          object.
        OSError: If resolving the reference or query directory fails, or
          collecting the top-level files of either directory fails.
        RuntimeError: If resolving the reference or query directory fails.
        ValueError: If the reference and query directories resolve to the same
          path, any filename is empty or not unique after sanitisation, any loaded
          BED file is invalid according to PyRanges, fewer than two reference BED
          files are loaded, or fewer than one query BED file is loaded.
    """

    if (
        resolve_bed_dir(bed_type='reference', bed_dir=bed_dirs['reference'])
        == resolve_bed_dir(bed_type='query', bed_dir=bed_dirs['query'])
    ):
        raise ValueError('--reference_dir and --query_dir resolve to the same path.')

    beds = {'reference': {}, 'query': {}}
    seen_names = {}

    for bed_type, bed_dir in bed_dirs.items():
        for path in collect_paths(bed_type, bed_dir):
            filename = sanitise_filename(path)
            check_filename_unique(path=path, filename=filename, seen_names=seen_names)
            seen_names[filename] = path
            
            beds[bed_type][filename] = read_valid_bed(path)

            logging.info(f'Loaded {bed_type} BED file from {path} as "{filename}".')

    if len(beds['reference']) < 2:
        raise ValueError('Received fewer than two reference BED files.')

    if len(beds['query']) < 1:
        raise ValueError('Received fewer than one query BED file.')

    return beds
