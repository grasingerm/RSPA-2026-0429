#!/usr/bin/env python3
"""Print states certified as stable in the full compatible configuration space.

This script post-processes the CSV files written by:

    degree6_stability_batch.py
    degree8_stability_batch.py

By default it looks for:

    degree6_stability_results/degree6_stability_summary.csv
    degree8_stability_results/degree8_stability_summary.csv

A state is printed when ``certified_relaxed_survivor`` is true. In the batch
scripts, this means that every tested finite-difference step classified the
state as a strict local minimum in the full compatible fold-angle space while
self-contact constraints were omitted.

Examples
--------
Run from the directory containing the two result folders:

    python print_fullspace_stable_states.py

Process explicitly specified folders or summary CSV files:

    python print_fullspace_stable_states.py \
        path/to/degree6_stability_results \
        path/to/degree8_stability_results

Print only the human-readable summary:

    python print_fullspace_stable_states.py --format summary

Print only Python-ready state lists and stable state indices:

    python print_fullspace_stable_states.py --format python

Also print every component of the full fold-angle vector:

    python print_fullspace_stable_states.py --show-full

Save the printed report while also displaying it:

    python print_fullspace_stable_states.py --output fullspace_stable_states.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_RESULT_PATHS = (
    Path("degree6_stability_results"),
    Path("degree8_stability_results"),
)

TRUE_STRINGS = {"1", "true", "t", "yes", "y"}
FALSE_STRINGS = {"0", "false", "f", "no", "n", ""}


@dataclass(frozen=True)
class StableState:
    """One state selected from a batch summary CSV."""

    degree: int
    source_file: Path
    state_id: str
    group_index: int
    state_index: int
    alpha1: float
    alpha1_over_pi: float
    rk: float
    phi: tuple[float, ...]
    consensus_status: str
    certified: bool
    closure_norm: float
    symmetry_norm: float
    full_gradient_norm: float
    breaking_gradient_norm: float
    minimum_full_eigenvalue: float
    minimum_breaking_eigenvalue: float
    status_by_step: str

    @property
    def compact_phi(self) -> tuple[float, ...]:
        """Return the independent symmetric fold angles used by the batch input."""
        if self.degree == 6:
            return self.phi[:4]
        if self.degree == 8:
            return self.phi[:3]
        return self.phi


@dataclass(frozen=True)
class FileResult:
    """Selected states and candidate count for one summary CSV."""

    source_file: Path
    degree: int
    total_candidates: int
    stable_states: tuple[StableState, ...]


def parse_bool(value: object, *, field_name: str) -> bool:
    """Parse a CSV boolean without treating the string 'False' as truthy."""
    text = "" if value is None else str(value).strip().lower()
    if text in TRUE_STRINGS:
        return True
    if text in FALSE_STRINGS:
        return False
    raise ValueError(f"Could not parse {field_name}={value!r} as a boolean.")


def parse_float(value: object, *, default: float = math.nan) -> float:
    """Parse a numeric CSV field, returning default for empty/missing values."""
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_int(value: object, *, default: int = -1) -> int:
    """Parse an integer-like CSV field."""
    number = parse_float(value)
    if not math.isfinite(number):
        return default
    return int(round(number))


def phi_column_number(name: str) -> int | None:
    """Return i for a column named phi<i>, otherwise None."""
    match = re.fullmatch(r"phi(\d+)", name.strip())
    return int(match.group(1)) if match else None


def infer_degree(summary_file: Path, row: dict[str, str]) -> int:
    """Infer vertex degree from the filename, then fall back to phi columns."""
    match = re.search(r"degree(\d+)", summary_file.name.lower())
    if match:
        return int(match.group(1))

    indices = [
        index
        for key in row
        if (index := phi_column_number(key)) is not None
        and str(row.get(key, "")).strip()
    ]
    if not indices:
        raise ValueError(
            f"Could not infer the vertex degree from {summary_file}; "
            "no degree tag or phi columns were found."
        )
    return max(indices)


def extract_phi(row: dict[str, str], degree: int) -> tuple[float, ...]:
    """Read phi1, ..., phiN from a batch summary row."""
    values: list[float] = []
    for index in range(1, degree + 1):
        key = f"phi{index}"
        value = parse_float(row.get(key))
        if not math.isfinite(value):
            raise ValueError(f"Missing or non-finite {key} in state {row.get('state_id')!r}.")
        values.append(value)
    return tuple(values)


def find_summary_files(paths: Sequence[Path]) -> list[Path]:
    """Resolve result folders or explicit summary CSV paths."""
    found: list[Path] = []

    for path in paths:
        expanded = path.expanduser()

        if expanded.is_file():
            if expanded.name.endswith("_stability_summary.csv"):
                found.append(expanded.resolve())
            else:
                print(
                    f"warning: skipping file that is not a stability summary CSV: "
                    f"{expanded}",
                    file=sys.stderr,
                )
            continue

        if expanded.is_dir():
            direct = sorted(expanded.glob("degree*_stability_summary.csv"))
            if direct:
                found.extend(item.resolve() for item in direct)
            else:
                recursive = sorted(expanded.rglob("degree*_stability_summary.csv"))
                found.extend(item.resolve() for item in recursive)
            continue

        print(f"warning: result path does not exist: {expanded}", file=sys.stderr)

    # Preserve discovery order while removing duplicates.
    unique: list[Path] = []
    seen: set[Path] = set()
    for item in found:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def row_is_selected(row: dict[str, str], criterion: str) -> bool:
    """Apply the requested stability-selection criterion."""
    consensus = str(row.get("consensus_status", "")).strip()

    if criterion == "certified":
        if "certified_relaxed_survivor" not in row:
            raise KeyError(
                "Summary CSV has no 'certified_relaxed_survivor' column. "
                "Use a summary written by the current batch scripts, or use "
                "--criterion consensus."
            )
        return parse_bool(
            row.get("certified_relaxed_survivor"),
            field_name="certified_relaxed_survivor",
        )

    if criterion == "consensus":
        return consensus == "strict_local_minimum"

    raise ValueError(f"Unknown selection criterion: {criterion!r}")


def read_summary_file(summary_file: Path, criterion: str) -> FileResult:
    """Read one batch summary and retain its full-space-stable states."""
    with summary_file.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {summary_file}")
        rows = list(reader)

    if not rows:
        match = re.search(r"degree(\d+)", summary_file.name.lower())
        degree = int(match.group(1)) if match else -1
        return FileResult(summary_file, degree, 0, ())

    degree = infer_degree(summary_file, rows[0])
    selected: list[StableState] = []

    for row in rows:
        if not row_is_selected(row, criterion):
            continue

        certified = (
            parse_bool(
                row.get("certified_relaxed_survivor"),
                field_name="certified_relaxed_survivor",
            )
            if "certified_relaxed_survivor" in row
            else False
        )

        selected.append(
            StableState(
                degree=degree,
                source_file=summary_file,
                state_id=str(row.get("state_id", "")).strip(),
                group_index=parse_int(row.get("group_index")),
                state_index=parse_int(row.get("state_index")),
                alpha1=parse_float(row.get("alpha1_rad")),
                alpha1_over_pi=parse_float(row.get("alpha1_over_pi")),
                rk=parse_float(row.get("rk")),
                phi=extract_phi(row, degree),
                consensus_status=str(row.get("consensus_status", "")).strip(),
                certified=certified,
                closure_norm=parse_float(row.get("closure_norm")),
                symmetry_norm=parse_float(row.get("symmetry_norm")),
                full_gradient_norm=parse_float(row.get("primary_full_gradient_norm")),
                breaking_gradient_norm=parse_float(
                    row.get("primary_breaking_gradient_norm")
                ),
                minimum_full_eigenvalue=parse_float(
                    row.get("primary_minimum_full_eigenvalue")
                ),
                minimum_breaking_eigenvalue=parse_float(
                    row.get("primary_minimum_breaking_eigenvalue")
                ),
                status_by_step=str(row.get("status_by_step", "")).strip(),
            )
        )

    selected.sort(key=lambda state: (state.group_index, state.state_index))
    return FileResult(summary_file, degree, len(rows), tuple(selected))


def format_number(value: float, precision: int) -> str:
    """Compact high-precision numeric formatting."""
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{precision}g}"


def format_tuple(values: Iterable[float], precision: int) -> str:
    """Format values as a Python-compatible tuple."""
    rendered = ", ".join(format_number(value, precision) for value in values)
    if rendered and "," not in rendered:
        rendered += ","
    return f"({rendered})"


def render_summary(
    results: Sequence[FileResult],
    *,
    precision: int,
    show_full: bool,
) -> list[str]:
    """Create a readable state-by-state report."""
    lines: list[str] = []
    grand_total = sum(result.total_candidates for result in results)
    grand_stable = sum(len(result.stable_states) for result in results)

    lines.append("FULL-SPACE STABLE STATES")
    lines.append("========================")
    lines.append(
        f"Selected {grand_stable} certified state(s) from "
        f"{grand_total} candidate state(s)."
    )
    lines.append(
        "State indices are one-based and match the ordering used in each "
        "batch CONFIGURATIONS list."
    )
    lines.append("")

    for result in results:
        lines.append(f"Source: {result.source_file}")
        lines.append(
            f"Degree {result.degree}: {len(result.stable_states)} stable "
            f"state(s) / {result.total_candidates} candidate(s)"
        )

        if not result.stable_states:
            lines.append("  No states satisfy the selected criterion.")
            lines.append("")
            continue

        groups: dict[int, list[StableState]] = defaultdict(list)
        for state in result.stable_states:
            groups[state.group_index].append(state)

        for group_index in sorted(groups):
            group = groups[group_index]
            first = group[0]
            indices = ", ".join(str(state.state_index) for state in group)
            lines.append(
                "  "
                f"Group {group_index}: "
                f"alpha1={format_number(first.alpha1, precision)} rad "
                f"({format_number(first.alpha1_over_pi, precision)}*pi), "
                f"rk={format_number(first.rk, precision)}"
            )
            lines.append(f"    stable state indices: [{indices}]")

            for state in group:
                lines.append(
                    f"    state {state.state_index} ({state.state_id})"
                )
                lines.append(
                    f"      compact symmetric angles = "
                    f"{format_tuple(state.compact_phi, precision)}"
                )
                if state.degree == 6:
                    lines.append(
                        "      Figure 3 coordinates "
                        f"(phi2, phi3) = "
                        f"{format_tuple((state.phi[1], state.phi[2]), precision)}"
                    )
                if show_full:
                    lines.append(
                        f"      full fold-angle vector = "
                        f"{format_tuple(state.phi, precision)}"
                    )
                lines.append(
                    "      diagnostics: "
                    f"closure={format_number(state.closure_norm, 6)}, "
                    f"symmetry={format_number(state.symmetry_norm, 6)}, "
                    f"|g_full|={format_number(state.full_gradient_norm, 6)}, "
                    f"|g_break|={format_number(state.breaking_gradient_norm, 6)}, "
                    f"lambda_min={format_number(state.minimum_full_eigenvalue, 6)}, "
                    f"lambda_break,min="
                    f"{format_number(state.minimum_breaking_eigenvalue, 6)}"
                )
                if state.status_by_step:
                    lines.append(f"      status by step: {state.status_by_step}")

        lines.append("")

    return lines


def render_python_literals(
    results: Sequence[FileResult],
    *,
    precision: int,
) -> list[str]:
    """Create Python-ready indices and compact CONFIGURATIONS lists."""
    lines: list[str] = []
    lines.append("PYTHON-READY OUTPUT")
    lines.append("===================")
    lines.append(
        "# State indices below are one-based, matching the batch output."
    )
    lines.append("")

    by_degree: dict[int, list[StableState]] = defaultdict(list)
    for result in results:
        by_degree[result.degree].extend(result.stable_states)

    for degree in sorted(by_degree):
        states = sorted(
            by_degree[degree],
            key=lambda state: (
                str(state.source_file),
                state.group_index,
                state.state_index,
            ),
        )
        variable_prefix = f"DEGREE{degree}_FULLSPACE_STABLE"

        grouped: dict[tuple[str, int], list[StableState]] = defaultdict(list)
        for state in states:
            grouped[(str(state.source_file), state.group_index)].append(state)

        lines.append(f"{variable_prefix}_INDICES_BY_GROUP = {{")
        for (_, group_index), group in sorted(
            grouped.items(), key=lambda item: (item[0][0], item[0][1])
        ):
            first = group[0]
            indices = ", ".join(str(state.state_index) for state in group)
            lines.append(
                f"    {group_index}: [{indices}],  "
                f"# alpha1/pi={format_number(first.alpha1_over_pi, precision)}, "
                f"rk={format_number(first.rk, precision)}"
            )
        lines.append("}")
        lines.append("")

        lines.append(f"{variable_prefix}_CONFIGURATIONS = [")
        for (_, group_index), group in sorted(
            grouped.items(), key=lambda item: (item[0][0], item[0][1])
        ):
            first = group[0]
            lines.append("    [")
            lines.append(
                f"        {format_number(first.alpha1, precision)},  "
                f"# alpha1 = {format_number(first.alpha1_over_pi, precision)} * pi; "
                f"group {group_index}"
            )
            lines.append(f"        {format_number(first.rk, precision)},  # rk")
            lines.append("        [")
            for state in group:
                lines.append(
                    f"            {format_tuple(state.compact_phi, precision)},  "
                    f"# state {state.state_index}"
                )
            lines.append("        ],")
            lines.append("    ],")
        lines.append("]")
        lines.append("")

    if not by_degree:
        lines.append("# No states satisfied the selected criterion.")
        lines.append("")

    return lines


def build_report(
    results: Sequence[FileResult],
    *,
    output_format: str,
    precision: int,
    show_full: bool,
) -> str:
    """Assemble the requested output sections."""
    sections: list[str] = []

    if output_format in {"summary", "both"}:
        sections.extend(
            render_summary(results, precision=precision, show_full=show_full)
        )

    if output_format == "both" and sections:
        sections.append("")

    if output_format in {"python", "both"}:
        sections.extend(render_python_literals(results, precision=precision))

    return "\n".join(sections).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "Result folders or explicit degree*_stability_summary.csv files. "
            "Defaults to degree6_stability_results and degree8_stability_results."
        ),
    )
    parser.add_argument(
        "--criterion",
        choices=("certified", "consensus"),
        default="certified",
        help=(
            "'certified' requires certified_relaxed_survivor=True at every "
            "finite-difference step (default). 'consensus' accepts rows whose "
            "consensus_status is strict_local_minimum."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("summary", "python", "both"),
        default="both",
        help="Choose human-readable output, Python-ready output, or both.",
    )
    parser.add_argument(
        "--show-full",
        action="store_true",
        help="Include the complete N-component fold-angle vector in the summary.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=17,
        help="Number of significant digits for fold angles and parameters.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional text file that receives the same report printed to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.precision < 6:
        raise SystemExit("--precision should be at least 6.")

    input_paths = tuple(args.paths) if args.paths else DEFAULT_RESULT_PATHS
    summary_files = find_summary_files(input_paths)

    if not summary_files:
        searched = ", ".join(str(path) for path in input_paths)
        raise SystemExit(
            "No degree*_stability_summary.csv files were found. "
            f"Searched: {searched}"
        )

    results: list[FileResult] = []
    failures: list[str] = []

    for summary_file in summary_files:
        try:
            results.append(read_summary_file(summary_file, args.criterion))
        except Exception as exc:
            failures.append(f"{summary_file}: {type(exc).__name__}: {exc}")

    if failures:
        for failure in failures:
            print(f"warning: {failure}", file=sys.stderr)

    if not results:
        raise SystemExit("No summary CSV could be read successfully.")

    results.sort(key=lambda result: (result.degree, str(result.source_file)))
    report = build_report(
        results,
        output_format=args.format,
        precision=args.precision,
        show_full=args.show_full,
    )

    print(report, end="")

    if args.output is not None:
        output_path = args.output.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"\nWrote report to: {output_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
