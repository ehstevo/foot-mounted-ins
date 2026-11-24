"""Run a bounded GNSS-aiding parameter sweep over immutable replay evidence.

Example:
    python scripts/sweep_gnss_zupt.py data/runs/<run>/imu.bmu \
        --gnss-fix-table data/runs/<run>/gnss_fixes.csv \
        --base-config config/gnss_ins_zupt/example.yaml \
        --plan config/tuning/gnss_example.yaml \
        --output-dir results/gnss-sweep

The plan declares one-at-a-time or Cartesian values for allowed GNSS-aiding
profile fields relative to one complete baseline configuration. Each generated
candidate receives a fresh raw-frame replay, resolved YAML profile, and detailed
GNSS-fusion diagnostics. The output directory also contains a cross-candidate
summary CSV and provenance manifest. Reported accepted-NIS statistics describe
the updates that passed the candidate's current gate; they are not direct GNSS
accuracy measurements.

Every candidate uses the same supplied IMU and GNSS evidence, but differing
filter settings change the estimator's internal state and gating history.
Consequently the sweep produces controlled descriptive evidence, not an
automatically optimal profile or a cross-candidate accuracy guarantee.
"""
import argparse
from pathlib import Path
import sys

from bootins.bmu.transport import iter_frames_replay
from bootins.reporting.gnss_fusion_export import write_gnss_fusion_diagnostics_csv
from bootins.sensors.gnss_table import load_gnss_fix_table
from bootins.analysis.gnss_tuning import (
    generate_gnss_sweep_candidates,
    load_gnss_sweep_plan,
    resolve_gnss_sweep_candidate,
    summarize_gnss_sweep_candidate,
)
from bootins.reporting.gnss_tuning_export import (
    write_gnss_sweep_manifest,
    write_gnss_sweep_summary_csv,
    write_resolved_gnss_sweep_profile,
)
from bootins.config.gnss_ins_zupt import (
    load_gnss_zupt_config_document,
    parse_gnss_zupt_config,
)
from bootins.replay.gnss_ins_zupt import run_gnss_zupt_replay


_DEFAULT_MAX_CANDIDATES = 32


def _positive_int(value: str) -> int:
    """Parse one positive command-line integer with argparse context."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _print_candidate_summary(candidate_index: int, candidate_count: int, summary) -> None:
    """Print compact non-ranking fusion evidence for one replay candidate.

    Args:
        candidate_index: One-based completed-candidate index.
        candidate_count: Total generated candidate count.
        summary: Candidate summary with outcome counts and accepted-NIS facts.
    """
    print(f"[{candidate_index}/{candidate_count}] {summary.candidate_name}")
    print(
        "  GNSS outcomes: "
        f"accepted={summary.accepted_count}, "
        f"nis-rejected={summary.rejected_count}, "
        f"stale={summary.stale_count}, "
        f"future-dropped={summary.future_dropped_count}"
    )
    print(
        "  Accepted NIS: "
        f"count={summary.accepted_nis.count}, "
        f"median={summary.accepted_nis.median}"
    )


def main() -> int:
    """Run every candidate in one declared GNSS-aiding sweep plan.

    Returns:
        ``0`` after all candidates and final artifacts are written, ``1`` when
        a candidate replay is rejected, or ``2`` when input validation, output
        creation, or artifact I/O fails.

    Notes:
        Each candidate starts a fresh replay from identical supplied evidence.
        The output directory contains resolved profiles, per-candidate fusion
        diagnostics, a summary CSV, and a sweep manifest; no automatic ranking
        or configuration selection is performed.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Replay one GNSS + INS + ZUPT evidence bundle for every "
            "candidate declared by a bounded filter-sweep plan."
        ),
    )
    parser.add_argument("capture_path", type=Path, help="Path to one .bmu capture")
    parser.add_argument(
        "--gnss-fix-table",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to one normalized GNSS fix-table CSV",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to the complete aided YAML profile used as the candidate base",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to one declared GNSS-aiding filter-sweep YAML plan",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="New directory for resolved profiles and sweep diagnostics",
    )
    parser.add_argument(
        "--max-candidates",
        type=_positive_int,
        default=_DEFAULT_MAX_CANDIDATES,
        metavar="COUNT",
        help=(
            "Maximum generated candidates before refusing the sweep "
            f"(default: {_DEFAULT_MAX_CANDIDATES})"
        ),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir must name a directory that does not yet exist.")

    try:
        base_document = load_gnss_zupt_config_document(args.base_config)
    except OSError as error:
        print(f"Could not read base GNSS profile {args.base_config}: {error}", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"Base GNSS profile rejected {args.base_config}: {error}", file=sys.stderr)
        return 2

    try:
        plan = load_gnss_sweep_plan(args.plan)
    except OSError as error:
        print(f"Could not read GNSS sweep plan {args.plan}: {error}", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"GNSS sweep plan rejected {args.plan}: {error}", file=sys.stderr)
        return 2

    try:
        candidates = generate_gnss_sweep_candidates(
            plan,
            max_candidates=args.max_candidates,
        )
    except ValueError as error:
        print(f"GNSS sweep plan rejected {args.plan}: {error}", file=sys.stderr)
        return 2

    resolved_candidates = []
    for candidate in candidates:
        try:
            resolved_document = resolve_gnss_sweep_candidate(base_document, candidate)
            config = parse_gnss_zupt_config(resolved_document)
        except ValueError as error:
            print(
                f"GNSS sweep candidate '{candidate.name}' rejected: {error}",
                file=sys.stderr,
            )
            return 2
        resolved_candidates.append((candidate, resolved_document, config))

    try:
        gnss_fixes = load_gnss_fix_table(args.gnss_fix_table)
    except OSError as error:
        print(
            f"Could not read GNSS fix table {args.gnss_fix_table}: {error}",
            file=sys.stderr,
        )
        return 2
    except ValueError as error:
        print(
            f"GNSS fix table rejected {args.gnss_fix_table}: {error}",
            file=sys.stderr,
        )
        return 2

    try:
        args.output_dir.mkdir(parents=True)
        profiles_dir = args.output_dir / "profiles"
        diagnostics_dir = args.output_dir / "diagnostics"
        profiles_dir.mkdir()
        diagnostics_dir.mkdir()
    except OSError as error:
        print(
            f"Could not create GNSS sweep output directory {args.output_dir}: {error}",
            file=sys.stderr,
        )
        return 2

    summaries = []
    candidate_count = len(resolved_candidates)
    for candidate_index, (candidate, resolved_document, config) in enumerate(
        resolved_candidates,
        start=1,
    ):
        profile_path = profiles_dir / f"{candidate.name}.yaml"
        diagnostics_path = diagnostics_dir / f"{candidate.name}.csv"
        try:
            write_resolved_gnss_sweep_profile(profile_path, resolved_document)
        except (OSError, ValueError) as error:
            print(
                f"Could not write GNSS sweep candidate '{candidate.name}': {error}",
                file=sys.stderr,
            )
            return 2

        try:
            replay_result = run_gnss_zupt_replay(
                iter_frames_replay(args.capture_path),
                gnss_fixes,
                config,
            )
            summary = summarize_gnss_sweep_candidate(candidate, replay_result)
        except OSError as error:
            print(
                f"Could not complete GNSS sweep candidate '{candidate.name}': {error}",
                file=sys.stderr,
            )
            return 2
        except ValueError as error:
            print(
                f"GNSS sweep candidate '{candidate.name}' rejected: {error}",
                file=sys.stderr,
            )
            return 1

        try:
            write_gnss_fusion_diagnostics_csv(
                diagnostics_path,
                replay_result.gnss_fusion_diagnostics,
            )
        except (OSError, ValueError) as error:
            print(
                f"Could not write GNSS sweep candidate '{candidate.name}': {error}",
                file=sys.stderr,
            )
            return 2
        summaries.append(summary)
        _print_candidate_summary(candidate_index, candidate_count, summary)

    try:
        write_gnss_sweep_summary_csv(args.output_dir / "summary.csv", summaries)
        write_gnss_sweep_manifest(
            args.output_dir / "sweep.meta.yaml",
            capture_path=args.capture_path,
            gnss_fix_table_path=args.gnss_fix_table,
            base_config_path=args.base_config,
            plan_path=args.plan,
            summaries=summaries,
        )
    except (OSError, ValueError) as error:
        print(
            f"Could not finalize GNSS sweep output {args.output_dir}: {error}",
            file=sys.stderr,
        )
        return 2

    print(f"Wrote GNSS sweep summary: {args.output_dir / 'summary.csv'}")
    print(f"Wrote GNSS sweep manifest: {args.output_dir / 'sweep.meta.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
