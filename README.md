# inteRelate

This package provides a full implementation of inteRelate. Here, we provide instructions that cover the installation, usage, and results of the software. For those looking to learn more about the theory and method behind inteRelate, an application note will be released soon. 

## 1. Installation

The commands below use `python`; if this command is unavailable, use `python3` on macOS or Linux, or `py` on Windows. inteRelate requires Python 3.12 or later.


### Users

Create a virtual environment `.venv` in the directory where you want to use inteRelate:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Alternatively, activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then upgrade pip and install inteRelate:

```bash
python -m pip install --upgrade pip
python -m pip install interelate
```

### Developers

Contributors should use an editable installation of inteRelate instead:

<details>

<summary>Developer instructions</summary>

Clone the repository:

```bash
git clone https://github.com/loggy01/interelate.git
cd interelate
```

Follow the user instructions above to create and activate `.venv` and upgrade pip. Instead of installing inteRelate from PyPI, install the repository in editable mode with its test dependencies:

```bash
python -m pip install -e ".[test]"
```

Run the test suite before and after making changes:

```bash
python -m pytest
```

Further information about the test suite is available [here](test/README.md).

</details>

## 2. Usage

To run inteRelate, activate its virtual environment and then execute the `interelate` command as follows:

```bash
interelate \
  --reference_dir=... \             # Required
  --query_dir=... \                 # Required
  --output_dir=... \                # Required
  --genomic_distances=... \         # Required
  --significance_level=... \        # Optional (Default = 0.05)          
  --yates_correction=... \          # Optional (Default = False)
  --power_divergence_lambda=... \   # Optional (Default = None)
  --resampling_method=... \         # Optional (Default = None)
  --association_statistic=... \     # Optional (Default = cramer)
  --pairwise_testing=... \          # Optional (Default = True)
  --adjust_method=... \             # Optional (Default = holm-sidak)
  --adjust_max_iterations=...       # Optional (Default = 1)
```

The `...` in the command are to be replaced by your own input. Here are your options:

<details>

<summary>Argument options</summary>

<dl>
  <dt><code>--reference_dir</code>:</dt>
  <dd>A path to a directory containing your reference BED files. At least two files must be found. Standard BED3 through BED12 files are accepted. The recognised file extensions are .bed, .bed.gz, .txt, and .txt.gz.
  </dd>

  <dt><code>--query_dir</code>:</dt>
  <dd>A path to a directory containing your query BED files. At least one file must be found. Standard BED3 through BED12 files are accepted. The recognised file extensions are .bed, .bed.gz, .txt, and .txt.gz.
  </dd>

  <dt><code>--output_dir</code>:</dt>
  <dd>A path to a directory in which to store inteRelate result files.
  </dd>

  <dt><code>--genomic_distances</code>:</dt>
  <dd>A comma-separated list of genomic distances (in bp) to use for counting overlaps between reference and query BED intervals. Please provide a unique list of non-negative integers without whitespace, e.g., <code>0,100,1000</code>.
  </dd>

  <dt><code>--significance_level</code>:</dt>
  <dd>An optional value between <code>0</code> and <code>1</code> to use for the significance level in χ² tests. The default is <code>0.05</code>.
  </dd>

  <dt><code>--yates_correction</code>:</dt>
  <dd>An optional value of <code>True</code> or <code>False</code> for whether to use Yates's correction in χ² and association tests. The default is <code>False</code>. It is forced to be <code>False</code> in global tests if there are more than two reference BED files.
  </dd>

  <dt><code>--power_divergence_lambda</code>:</dt>
  <dd>An optional floating point value to use as a statistic from the Cressie-Read power divergence family in place of Pearson's χ² statistic in χ² and association tests. The default is <code>None</code>.
  </dd>

  <dt><code>--resampling_method</code>:</dt>
  <dd>An optional value of <code>permutation</code> or <code>monte_carlo</code> to determine which resampling method to use in χ² tests. The defualt is <code>None</code>. If given, an error is raised if <code>--power_divergence_lambda</code> is not <code>None</code> and/or <code>--yates_correction</code> is <code>True</code>.
  </dd>

  <dt><code>--association_statistic</code>:</dt>
  <dd>An optional value of <code>cramer</code>, <code>tschuprow</code>, or <code>pearson</code> to determine which association statistic to calculate. The default is <code>cramer</code>.
  </dd>

  <dt><code>--pairwise_testing</code>:</dt>
  <dd>An optional value of <code>True</code> or <code>False</code> for whether to run pairwise testing between reference BED files. The default is <code>True</code>. It is forced to be <code>False</code> if there are only two reference BED files or in the event of a non-significant global χ² test result.
  </dd>

  <dt><code>--adjust_method</code>:</dt>
  <dd>An optional value of <code>bonferroni</code>, <code>sidak</code>, <code>holm-sidak</code>, <code>holm</code>, <code>simes-hochberg</code>, <code>hommel</code>, <code>fdr_bh</code>, <code>fdr_by</code>, <code>fdr_tsbh</code>, or <code>fdr_tsbky</code> to determine which p-value adjustment method to use in multiple testing correction. The default is <code>holm-sidak</code>.
  </dd>

  <dt><code>--adjust_max_iterations</code>:</dt>
  <dd>An optional integer value to use as the maximum number of iterations to perform when using two-stage FDR for <code>--adjust_method</code>. The default value is <code>1</code>. A value of <code>-1</code> corresponds to full iterations, which is equal to the number of pairwise tests performed. A value of <code>0</code> uses only a single-stage FDR adjustment using a <code>bh</code> or <code>bky</code> prior fraction of assumed true hypotheses. It is ignored if the <code>--adjust_method</code> is not <code>fdr_tsbh</code> or <code>fdr_tsbky</code>.
  </dd>
</dl>

</details>

## 3. Results

inteRelate stores the results of a successful run within the user-provided output directory. For example, if a user provided three reference BED files (`ref1.bed`, `ref2.bed`, and `ref3.bed`), two query BED files (`query1.bed` and `query2.bed`), the output directory `output`, and the genomic distances `100` and `1000`, the results would have the following structure:

```text
output/
├── interelate.log
├── overlap_counts/
│   ├── query1_ref1.txt
│   ├── query1_ref2.txt
│   ├── query1_ref3.txt
│   ├── query2_ref1.txt
│   ├── query2_ref2.txt
│   └── query2_ref3.txt
├── query1_1000bp.json
├── query1_100bp.json
├── query2_1000bp.json
└── query2_100bp.json
```

> [!WARNING]
> BED file names and genomic distances are used to name the result files. Therefore, repeated runs with the same input will lead to the overwriting of result files.

The results structure comprises three parts:

<details>

<summary>Results structure</summary>

1. An individual `interelate.log` file is made, which stores all the output messages of the run, including records of the user's input choices.

2. A subdirectory called `overlap_counts` is made, which contains a .txt file for each query-reference pair. Each of these files are the same as their parental reference BED file, but with columns added on to the end representing the overlap count at each given genomic distance. In this example, a column each would be added for 100 bp and 1000 bp, representing how many query intervals overlap each reference interval within the given distance for that pair.

3. A .json file is made for each query-distance pair, representing the statistical results for that query at that genomic distance. The .json files have a custom design to allow for the description of the statistcal results in a unified and human-readable format. In this example, a .json file would look like this (again, `...` are placeholders for real values):

    ```json
    {
      "overlap_result": {
        "observed_frequencies": {
          "overlap": [
            ["ref1", ...],
            ["ref2", ...],
            ["ref3", ...]
          ],
          "no_overlap": [
            ["ref1", ...],
            ["ref2", ...],
            ["ref3", ...]
          ]
        },
        "overlap_rate": [
          ["ref1", ...],
          ["ref2", ...],
          ["ref3", ...]
        ]
      },

      "global_testing_result": {
        "expected_frequencies": {
          "overlap": [
            ["ref1", ...],
            ["ref2", ...],
            ["ref3", ...]
          ],
          "no_overlap": [
            ["ref1", ...],
            ["ref2", ...],
            ["ref3", ...]
          ]
        },
        "chi2_statistic": ...,
        "dof": ...,
        "p_value": ...,
        "reject_null": ...,
        "association_statistic": ...
      },

      "pairwise_testing_result": {
        "expected_frequencies": {
          "overlap": [
            [["ref1", ...], ["ref2", ...]],
            [["ref1", ...], ["ref3", ...]],
            [["ref2", ...], ["ref3", ...]]
          ],
          "no_overlap": [
            [["ref1", ...], ["ref2", ...]],
            [["ref1", ...], ["ref3", ...]],
            [["ref2", ...], ["ref3", ...]]
          ]
        },
        "chi2_statistic": [
          [["ref1", "ref2"], ...],
          [["ref1", "ref3"], ...],
          [["ref2", "ref3"], ...]
        ],
        "p_value": [
          [["ref1", "ref2"], ...],
          [["ref1", "ref3"], ...],
          [["ref2", "ref3"], ...]
        ],
        "adjusted_p_value": [
          [["ref1", "ref2"], ...],
          [["ref1", "ref3"], ...],
          [["ref2", "ref3"], ...]
        ],
        "reject_null": [
          [["ref1", "ref2"], ...],
          [["ref1", "ref3"], ...],
          [["ref2", "ref3"], ...]
        ],
        "association_statistic": [
          [["ref1", "ref2"], ...],
          [["ref1", "ref3"], ...],
          [["ref2", "ref3"], ...]
        ]
      }
    }
    ```

    The .json file itself is comprised of three parts:

    <ol type="i">
      <li><code>overlap_result</code> reports a contingency table containing, for each reference, the number of intervals that do and don't overlap with at least one query interval. Additionally, this is simplified into an overlap rate for each reference.
      </li>

      <li><code>global_testing_result</code> reports a χ² test result and association statistic derived from the whole contingency table. In instances when all references show complete overlap or no overlap, tests cannot be performed and this section will be reported as null.
      </li>

      <li><code>pairwise_testing_result</code> reports χ² test results, adjusted p-values, and association statistics derived from reference pairs in the contingency table. In instances when the global χ² test gave an insignificant p-value, is reported as null, or pairwise testing is switched off, this section will be reported as null. Additionally, if both references in a pair show complete overlap or no overlap, results specific to that pair will be reported as null.
      </li>
    </ol>

</summary>