#!/usr/bin/env python3
"""
Run all 8 PCA variants sequentially, regress, and export results to Excel.

For each variant:
  1. Run PCA script  → overwrites final/pca/pca_scores_combined_tui.csv
  2. Run firstday regression  (--ife --pca --pltfe)
  3. Run QA regression        (--ife --pca --pltfe)
  4. Run every regression     (--rc1 --rc2 --no-rc3 --ic --pc --year-fe --ind-fe
                                --pltfe --no-winsor-x --pca --top-rivals N)
     for each N in --top-rivals (default: 1; full run: 1,3,5,10)
  5. Export every PCA tables  → reg_tables_every_pca.xlsx
  6. Copy Excels to variant-suffixed names in final/reg/

Usage:
  python final/pca/run_pca_variants.py [--variants base fer_3ratios ...]
  python final/pca/run_pca_variants.py --top-rivals 1,3,5,10
  python final/pca/run_pca_variants.py --list
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent.parent
PCA_DIR = ROOT / "final" / "pca"
REG_DIR = ROOT / "final" / "reg"
PYTHON  = sys.executable

ALL_VARIANTS = [
    ("base",          "pca_combined_tui.py"),
    ("fer_3ratios",   "pca_combined_tui_fer_3ratios.py"),
    ("fer_8emos",     "pca_combined_tui_fer_8emos.py"),
    ("fer_all",       "pca_combined_tui_fer_all.py"),
    ("gaze_reduced",  "pca_combined_tui_gaze_reduced.py"),
    ("gaze_3ratios",  "pca_combined_tui_gaze_3ratios.py"),
    ("gaze_8emos",    "pca_combined_tui_gaze_8emos.py"),
    ("gaze_all",      "pca_combined_tui_gaze_all.py"),
]

EVERY_BASE_ARGS = [
    "--rc1", "--rc2", "--no-rc3", "--ic", "--pc",
    "--year-fe", "--ind-fe", "--pltfe", "--no-winsor-x", "--pca",
    "--a-pros-qa-mod", "--max-pcs", "5",
]

MEAN_ARGS = ["--ife", "--pca", "--pltfe"]


def _parse_args():
    p = argparse.ArgumentParser(description="Run PCA variants → regression → export")
    p.add_argument("--variants", nargs="+", default=None,
                   help="Subset of variant names to run (default: all 8)")
    p.add_argument("--list", action="store_true", help="List available variants and exit")
    p.add_argument("--skip-on-error", action="store_true",
                   help="Log errors and continue instead of aborting")
    p.add_argument("--top-rivals", default="1",
                   help="Comma-separated top-rivals values for every regression (default: 1)")
    return p.parse_args()


def run(cmd, label="", skip_on_error=False):
    tag = label or " ".join(str(c) for c in cmd[-3:])
    print(f"\n--- {tag}")
    result = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    if result.returncode != 0:
        msg = f"ERROR (exit {result.returncode}): {tag}"
        if skip_on_error:
            print(msg)
            return False
        print(msg)
        sys.exit(result.returncode)
    return True


def run_parallel(jobs, skip_on_error=False):
    """Launch all (cmd, label) jobs simultaneously, wait for all to finish."""
    procs = []
    for cmd, label in jobs:
        tag = label or " ".join(str(c) for c in cmd[-3:])
        print(f"\n--- {tag} [parallel]")
        p = subprocess.Popen([str(c) for c in cmd], cwd=ROOT)
        procs.append((p, tag))

    ok = True
    for p, tag in procs:
        p.wait()
        if p.returncode != 0:
            msg = f"ERROR (exit {p.returncode}): {tag}"
            print(msg)
            if not skip_on_error:
                # kill remaining
                for other, _ in procs:
                    if other.poll() is None:
                        other.terminate()
                sys.exit(p.returncode)
            ok = False
    return ok


def main():
    _args = _parse_args()

    if _args.list:
        for name, script in ALL_VARIANTS:
            print(f"  {name:<18} {script}")
        return

    variants = ALL_VARIANTS
    if _args.variants:
        name_set = set(_args.variants)
        variants = [(n, s) for n, s in ALL_VARIANTS if n in name_set]
        missing  = name_set - {n for n, _ in variants}
        if missing:
            print(f"Unknown variants: {missing}")
            sys.exit(1)

    skip = _args.skip_on_error
    top_rivals = [int(x) for x in _args.top_rivals.split(",")]

    for variant, pca_script in variants:
        print(f"\n{'#' * 70}")
        print(f"  VARIANT: {variant}  ({pca_script})")
        print(f"{'#' * 70}")

        # 1. PCA
        ok = run([PYTHON, PCA_DIR / pca_script], f"PCA {variant}", skip)
        if not ok:
            continue

        # Save variant-specific copies of PCA outputs
        for stem in ("pca_loadings_combined_tui", "pca_explained_variance_combined_tui"):
            src = PCA_DIR / f"{stem}.csv"
            dst = PCA_DIR / f"{stem}_{variant}.csv"
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  Saved → {dst.name}")
            else:
                print(f"  WARNING: {src.name} not found, skipping copy")

        # 2-5. All regressions in parallel
        reg_jobs = [
            ([PYTHON, REG_DIR / "reg_bivariate_grouped_mean_firstday.py"] + MEAN_ARGS,
             "reg firstday"),
            ([PYTHON, REG_DIR / "reg_bivariate_grouped_mean_qa.py"] + MEAN_ARGS,
             "reg QA"),
        ] + [
            ([PYTHON, REG_DIR / "reg_bivariate_grouped_every.py"]
             + EVERY_BASE_ARGS + ["--top-rivals", str(n)],
             f"reg every (top{n})")
            for n in top_rivals
        ]
        ok = run_parallel(reg_jobs, skip)
        if not ok:
            continue

        # 6. Export tables
        run([PYTHON, REG_DIR / "export_reg_tables_pca_main.py"], "export main PCA", skip)

        # 7. Copy Excel output to variant-specific name
        for stem in ("reg_tables_pca_main",):
            src = REG_DIR / f"{stem}.xlsx"
            dst = REG_DIR / f"{stem}_{variant}.xlsx"
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  Saved → {dst.name}")
            else:
                print(f"  WARNING: {src.name} not found, skipping copy")

    print(f"\n{'#' * 70}")
    print("  ALL VARIANTS COMPLETE")
    print(f"{'#' * 70}")
    for variant, _ in variants:
        for stem in ("reg_tables_pca_main",):
            p = REG_DIR / f"{stem}_{variant}.xlsx"
            mark = "OK" if p.exists() else "MISSING"
            print(f"  [{mark}] {p.name}")


if __name__ == "__main__":
    main()
