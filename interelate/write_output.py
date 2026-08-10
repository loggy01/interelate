"""Writes each query overlap counts and each query formatted statistical result
  to file, as well as the interelate log file."""

import json
import logging
from logging.handlers import MemoryHandler
from pathlib import Path
from typing import Any

from interelate.calculate_overlap_counts import OverlapCounts
from interelate.format_statistical_results import FormattedStatisticalResults

# Handler to store log records in memory until they are flushed to the log file.
RUN_LOG_HANDLER = MemoryHandler(capacity=1000000, flushLevel=logging.CRITICAL + 1)

# JSON-like container types used by the custom formatter.
CONTAINERS = (dict, list, tuple)
SEQUENCES = (list, tuple)


def write_overlap_counts(
    overlap_counts: OverlapCounts,
    output_dir: Path
) -> None:
    """Writes each query overlap count to file."""

    overlap_counts_subdir = output_dir / 'overlap_counts'
    overlap_counts_subdir.mkdir(exist_ok=True)

    for query_name, query_overlap_counts in overlap_counts.items():
        for reference_name, overlap_count in query_overlap_counts.items():
            output_path = overlap_counts_subdir / f'{query_name}_{reference_name}.txt'
            overlap_count.to_csv(output_path, sep='\t', index=False)


def is_scalar_pair(value: Any) -> bool:
    """Returns whether a value is a scalar pair."""

    if not isinstance(value, SEQUENCES) or len(value) != 2:
        is_scalar_pair_value = False
    else:
        is_scalar_pair_value = all(
            not isinstance(item, CONTAINERS)
            for item in value
        )

    return is_scalar_pair_value


def is_comparison_value_pair(value: Any) -> bool:
    """Returns whether a value is a comparison/value pair."""

    if not isinstance(value, SEQUENCES) or len(value) != 2:
        is_comparison_value_pair_value = False
    else:
        is_comparison_value_pair_value = (
            is_scalar_pair(value[0])
            and not isinstance(value[1], CONTAINERS)
        )

    return is_comparison_value_pair_value


def is_expected_frequency_row(value: Any, level: int) -> bool:
    """Returns whether a value is a pairwise expected-frequency row."""

    if level < 4 or not isinstance(value, SEQUENCES):
        is_expected_frequency_row_value = False
    else:
        is_expected_frequency_row_value = all(
            is_scalar_pair(item)
            for item in value
        )

    return is_expected_frequency_row_value


def format_json(value: Any, level: int = 0) -> str:
    """Recursively formats JSON with custom indentation and line breaks."""

    indent = 2
    current_indent = ' ' * indent * level
    next_indent = ' ' * indent * (level + 1)

    if isinstance(value, dict):
        if not value:
            formatted_json_text = '{}'
        else:
            separator = ',\n\n' if level == 0 else ',\n'
            formatted_items = separator.join(
                f'{next_indent}{json.dumps(key)}: {format_json(item, level + 1)}'
                for key, item in value.items()
            )
            formatted_json_text = '{\n' + formatted_items + f'\n{current_indent}}}'

        return formatted_json_text

    if isinstance(value, SEQUENCES):
        if not value:
            formatted_json_text = '[]'
        elif (
            is_scalar_pair(value)
            or is_comparison_value_pair(value)
            or is_expected_frequency_row(value, level)
        ):
            formatted_json_text = json.dumps(value)
        else:
            formatted_items = ',\n'.join(
                f'{next_indent}{format_json(item, level + 1)}'
                for item in value
            )
            formatted_json_text = '[\n' + formatted_items + f'\n{current_indent}]'

        return formatted_json_text

    formatted_json_text = json.dumps(value)

    return formatted_json_text


def write_statistical_results(
    formatted_statistical_results: FormattedStatisticalResults,
    output_dir: Path
) -> None:
    """Writes each query formatted statistical result to file."""

    for query_name, query_formatted_statistical_results in (
        formatted_statistical_results.items()
    ):
        for genomic_distance, query_formatted_statistical_result in (
            query_formatted_statistical_results.items()
        ):
            output_path = (output_dir / f'{query_name}_{genomic_distance}.json')

            with output_path.open('w') as f:
                f.write(format_json(query_formatted_statistical_result))
                f.write('\n')


def write_log_file(output_dir: Path) -> None:
    """Writes the Interelate run log file."""

    logging.info('Done! Results written to %s', output_dir)

    output_path = output_dir / 'interelate.log'

    with output_path.open('w') as f:
        run_log_handler = logging.StreamHandler(f)
        run_log_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

        for record in RUN_LOG_HANDLER.buffer:
            run_log_handler.handle(record)


def write_output(
    overlap_counts: OverlapCounts,
    formatted_statistical_results: FormattedStatisticalResults,
    output_dir: Path
) -> None:
    """Writes each output to file.
    
    Args:
        overlap_counts: Dictionary containing the overlap counts between each
          query BED file and each reference BED file at each genomic distance in 
          OverlapCounts format.
        formatted_statistical_results: Dictionary containing the formatted
          statistical results for each query raw statistical result in 
          FormattedStatisticalResults format.
        output_dir: Path to the directory where all output files will be 
          written.
    
    Returns:
        None.
    """

    write_overlap_counts(
        overlap_counts=overlap_counts,
        output_dir=output_dir
    )

    write_statistical_results(
        formatted_statistical_results=formatted_statistical_results,
        output_dir=output_dir
    )

    write_log_file(output_dir=output_dir)
