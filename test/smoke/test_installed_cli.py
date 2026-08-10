import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import PipelineBedDirectoriesFactory

pytestmark = pytest.mark.smoke


def find_interelate_executable() -> Path:
    executable_on_path = shutil.which('interelate')
    candidates = []

    if executable_on_path is not None:
        candidates.append(Path(executable_on_path))

    candidates.extend(
        [
            Path(sys.executable).with_name('interelate'),
            Path(sys.executable).with_name('interelate.exe')
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise AssertionError('The installed interelate command was not found.')


def test_installed_cli_completes_a_minimal_run(
    pipeline_bed_directories_factory: PipelineBedDirectoriesFactory,
    tmp_path: Path
) -> None:
    bed_directories = pipeline_bed_directories_factory(
        {'query.bed': ((1, 1), (1, 1))},
        ('reference_a.bed', 'reference_b.bed')
    )
    output_dir = tmp_path / 'command-output'
    executable = find_interelate_executable()
    reference_dir = bed_directories['reference']
    query_dir = bed_directories['query']

    result = subprocess.run(
        [
            str(executable),
            f'--reference_dir={reference_dir}',
            f'--query_dir={query_dir}',
            f'--output_dir={output_dir}',
            '--genomic_distances=0',
            '--nopairwise_testing'
        ],
        capture_output=True,
        text=True,
        check=False
    )

    assert result.returncode == 0
    assert result.stderr == ''
    assert {
        path.name
        for path in output_dir.iterdir()
    } == {
        'interelate.log',
        'overlap_counts',
        'query_0bp.json'
    }
    assert {
        path.name
        for path in (output_dir / 'overlap_counts').iterdir()
    } == {
        'query_reference_a.txt',
        'query_reference_b.txt'
    }

    statistical_result = json.loads(
        (output_dir / 'query_0bp.json').read_text(encoding='utf-8')
    )
    assert statistical_result['overlap_result'][
        'observed_frequencies'
    ] == {
        'overlap': [
            ['reference_a', 1],
            ['reference_b', 1]
        ],
        'no_overlap': [
            ['reference_a', 1],
            ['reference_b', 1]
        ]
    }
    assert statistical_result['global_testing_result'][
        'reject_null'
    ] is False
    assert statistical_result['pairwise_testing_result'] is None
