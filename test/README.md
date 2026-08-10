# Test suite

Here, we provide the philosophy behind the inteRelate test suite in the hopes that future developers can understand the decisions taken and improve upon them.

## 1. Unit tests

Unit tests check package-owned behaviour at the lowest useful level. Each test uses the smallest practical input that still contains all the cases needed to demonstrate the behaviour being examined, avoiding unrelated data that could obscure a failure. Exact assertions identify precisely which decision or transformation changed.

When a function repeats work across named dimensions, such as every query-reference or query-distance pair, its tests normally cover three responsibilities separately: the lower-level calculation for one pair; organisation, meaning that results are stored under the correct names in the returned structure; and delegation, meaning that the lower-level function receives the correct objects and arguments once for every required pair.

Unit tests use real in-memory objects and real temporary files when the calculation or filesystem behaviour is itself under test. Mocks and monkeypatches are used for orchestration, delegation, configuration forwarding, and forced error paths. They do not replace the behaviour that the test is intended to verify.

### `test_cli.py`

Checks genomic distance filtering, output directory creation and reuse, default and non-default statistical configuration, statistical resampling method construction, incompatible statistical options, pairwise testing decisions, Yates's correction warnings, p-value adjustment method and iteration handling, CLI logging, required flags, and pipeline dispatch.

This file does not follow the calculation-organisation-delegation structure because `cli.py` is a command-line adapter rather than a function that processes repeated combinations. Its tests inject controlled, already-parsed flag values and replace the loading and pipeline functions when checking dispatch. This isolates inteRelate's CLI decisions without retesting Abseil's command-line parser, which is exercised through the installed command by the smoke test.

### `test_load_beds.py`

Checks directory resolution, file discovery, supported and compressed extensions, filename sanitisation and uniqueness, BED validation, loading errors, exact loaded contents, organisation by BED type and filename, and delegation across every BED type-file pair.

This file adapts the standard structure because `load_beds.py` contains several sequential filesystem, filename, reading, and validation helpers. Successful loading is tested using real temporary BED files and real PyRanges reading. Filesystem and reader operations are replaced only to force error paths or to check exact top-level delegation. Each helper is therefore tested separately before the tests check organisation and delegation.

### `test_run_pipeline.py`

Checks that each pipeline stage runs in order, receives the exact output of the preceding stage and the correct configuration, and emits the expected progress messages.

This file does not require separate calculation and organisation tests because `run_pipeline.py` is a linear orchestrator rather than a calculation or nested-combination function. Each pipeline stage is replaced with a mock so that the test can supply a unique result, trace its movement through the pipeline, and verify the exact order and arguments. The real stage behaviour is covered by its corresponding unit tests and by the integration tests.

### `test_calculate_overlap_counts.py`

Checks exact overlap counts for each genomic distance, preservation of input BEDs, organisation by query and reference name, and delegation across every query-reference pair.

This file follows the standard structure directly. Real PyRanges objects are used to test the lower-level calculation and result organisation. The lower-level calculation function is replaced only in the delegation test so that every query-reference call and returned result can be identified exactly.

### `test_build_contingency_tables.py`

Checks exact overlap and no-overlap frequencies, organisation by query and distance name, and delegation across every query-distance pair.

This file follows the standard structure directly. Real overlap-count PyRanges objects are used to test the lower-level calculation and result organisation. The lower-level table-building function is replaced only in the delegation test so that every query-distance call and returned table can be identified exactly.

### `test_run_statistical_testing.py`

Checks significant, insignificant, and invalid global tests; exact statistical results; configuration forwarding; Yates's correction; valid and invalid pairwise comparisons; multiple testing adjustment; pairwise execution decisions; result organisation; and delegation across every query-distance pair.

This file extends the standard structure because the production module contains global, pairwise, one-query, and all-query testing layers. Real NumPy arrays and statistical functions are used when checking scientific outcomes. Statistical functions are replaced only when checking configuration forwarding, multiple-testing input and output mapping, or delegation between layers. The tests cover each statistical layer and its possible outcomes before checking top-level organisation and delegation.

### `test_format_statistical_results.py`

Checks formatting of significant, insignificant, and invalid global outcomes; valid and invalid pairwise outcomes; Python-native output values; present and absent pairwise results; result organisation; and delegation across every query-distance pair.

This file extends the standard structure because the production module separately formats global, pairwise, one-query, and all-query results. Controlled raw result structures containing real NumPy and statistical values are used to test formatting. Lower-level formatting functions are replaced only when checking delegation between formatting layers. The tests therefore cover each formatting layer and structural outcome before checking top-level organisation and delegation.

### `test_write_output.py`

Checks exact overlap-count filenames and contents, JSON formatting and helper classifications, every query-reference overlap file, every query-distance JSON file, overwriting existing files, buffered log output, and delegation across all output-writing stages.

This file adapts the standard structure because `write_output.py` contains independent classification, formatting, file-writing, and orchestration functions rather than one lower-level calculation. Real temporary directories and files are used for all writing, formatting, logging, and overwriting behaviour. Writer functions are replaced only in the final orchestration test so that their order and exact arguments can be checked.

## 2. Integration tests

Integration tests use real temporary input files and run connected package stages without replacing internal functions. They cover representative cross-stage outcomes, including broader consequences that cannot be established by one unit alone. Warnings and errors are included only when they demonstrate an important effect across those connected stages.

The tests inject controlled, already-parsed CLI flag values and begin at `cli.main()`. This deliberately excludes Abseil's process-level parsing while retaining all inteRelate CLI decisions. From that point onward, BED loading, overlap calculation, contingency-table construction, statistical testing, formatting, writing, and logging all use their real implementations.

### `test_pipeline.py`

Runs `cli.main()` with real temporary BED files and all real internal stages. It covers invalid and insignificant global results, significant results with pairwise testing disabled, significant and insignificant pairwise comparisons, invalid pairwise comparisons, two-reference behaviour, overwriting on rerun, and propagation of a malformed-BED error. It checks the corresponding semantic results, output filenames, selected overlap output, warnings, and logging.

The frequency-table fixture generates small but genuine BED files whose overlaps produce the required statistical scenarios. These files are controlled test data, not replacements for the file loading or overlap implementation. No internal package function is mocked in this test file.

## 3. Smoke tests

Smoke tests run the installed command through its real entry point with a minimal valid input. They check that packaging, command-line parsing, basic execution, public output names, and a few semantic results are connected correctly. Complete output details, branches, and scientific behaviour remain with the unit and integration tests.

Smoke tests do not inject parsed flag values or replace package functions. They start a separate process containing the installed executable, the real Abseil parser, every real pipeline stage, and real filesystem access.

### `test_installed_cli.py`

Runs the installed `interelate` executable using real command-line flags and temporary BED files. It checks successful execution, clean standard error, exact public output filenames, valid JSON, exact overlap frequencies, the global decision, and the absence of pairwise results for one minimal run.

The BED files are small synthetic inputs generated for the test, but they are read and processed as genuine user files. No package component is mocked or monkeypatched.

## 4. What we do not test

The suite does not exhaustively retest behaviour owned by dependencies. Abseil is responsible for its general flag-parsing rules, while SciPy and statsmodels are responsible for their statistical implementations. The unit tests instead check every choice that inteRelate handles differently and verify that other values are forwarded correctly.

The integration and smoke tests do not repeat every unit-level edge case or every combination of command-line options. They cover representative cross-stage and installed-command journeys because repeating the complete unit matrix would make the suite slower, more brittle, and harder to diagnose.

Fixed byte-for-byte JSON and TSV expectations are limited to unit tests where formatting is the behaviour under test. Integration tests compare parsed semantic results; their rerun scenario compares files only with the preceding valid run to prove overwriting. The smoke test checks only a few decisive values.

Warnings and errors are not repeated at every level. Unit tests cover the package-owned branches and messages in detail. Integration tests include only failures or warnings with an important cross-stage effect, and the smoke test remains a successful minimal run.

Mocks are not used merely to make real package behaviour easier to reproduce. Where a real calculation, file operation, or statistical result can be tested clearly and deterministically, the real implementation is preferred. Internal package functions are never replaced in the integration or smoke tests.

Individual tests do not simulate different operating systems. Cross-platform support should instead be checked by running the suite on every supported operating system in continuous integration.

Tests are not added solely to reach a coverage percentage. Coverage helps find untested code, but the objective is confidence in package behaviour.

## 5. Why there is no functional section

Functional testing describes the purpose of checking user-visible behaviour, rather than a separate level in this suite. The integration tests provide the main functional scenario coverage, while the smoke test supplies one black-box journey through the installed command. A separate functional section would repeat those responsibilities without adding a distinct test boundary.

A separate functional section would become appropriate if several user-facing workflows needed to be tested specifically through the installed process boundary and could not be established by the integration tests. Examples include package-owned exit-status or terminal-output contracts, complex interactions between real command-line options, installed-resource discovery, or several distinct installed-command workflows. At that point, the smoke test should remain minimal and the substantive black box journeys should move to a functional section.