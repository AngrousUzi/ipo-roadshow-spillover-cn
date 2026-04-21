#!/usr/bin/env python3
"""
Run EFA variants sequentially, regress, and export results to Excel.

For each variant:
  1. Run EFA script  → overwrites final/pca/efa_scores_combined_tui.csv
  2. Run firstday regression  (--ife --efa --pltfe)
  3. Run QA regression        (--ife --efa --pltfe)
  4. Run every regression     (--rc1 --rc2 --no-rc3 --ic --pc --year-fe --ind-fe
                                --pltfe --no-winsor-x --efa --a-pros-qa-mod)
     for each N in --top-rivals (default: 1)
  5. Export EFA tables  → reg_tables_efa_main.xlsx
  6. Copy outputs to variant-suffixed names in final/pca/ and final/reg/

Usage:
  python final/pca/run_efa_variants.py
  python final/pca/run_efa_variants.py --variants base oblimin
  python final/pca/run_efa_variants.py --top-rivals 1,3,5,10
  python final/pca/run_efa_variants.py --list
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

# Each variant: (name, extra_args_to_efa_script)
ALL_VARIANTS = [
    ("base",    []),
    ("oblimin", ["--rotation", "oblimin"]),
    ("none",    ["--rotation", "none"]),
]

EVERY_BASE_ARGS = [
    "--rc1", "--rc2", "--no-rc3", "--ic", "--pc",
    "--year-fe", "--ind-fe", "--pltfe", "--no-winsor-x", "--efa",
    "--a-pros-qa-mod",
]

MEAN_ARGS = ["--ife", "--efa", "--pltfe"]


def _parse_args():
    p = argparse.ArgumentParser(description="Run EFA variants → regression → export")
    p.add_argument("--variants", nargs="+", default=None,
                   help="Subset of variant names to run (default: all)")
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
                for other, _ in procs:
                    if other.poll() is None:
                        other.terminate()
                sys.exit(p.returncode)
            ok = False
    return ok


def main():
    _args = _parse_args()

    if _args.list:
        for name, extra in ALL_VARIANTS:
            extra_str = " ".join(extra) if extra else "(default)"
            print(f"  {name:<12} {extra_str}")
        return

    variants = ALL_VARIANTS
    if _args.variants:
        name_set = set(_args.variants)
        variants = [(n, e) for n, e in ALL_VARIANTS if n in name_set]
        missing  = name_set - {n for n, _ in variants}
        if missing:
            print(f"Unknown variants: {missing}")
            sys.exit(1)

    skip = _args.skip_on_error
    top_rivals = [int(x) for x in _args.top_rivals.split(",")]

    for variant, efa_extra_args in variants:
        print(f"\n{'#' * 70}")
        print(f"  EFA VARIANT: {variant}  (rotation={efa_extra_args or 'varimax'})")
        print(f"{'#' * 70}")

        # 1. Run EFA
        ok = run(
            [PYTHON, PCA_DIR / "efa_combined_tui.py"] + efa_extra_args,
            f"EFA {variant}", skip,
        )
        if not ok:
            continue

        # Save variant-specific copies of EFA outputs
        for stem in ("efa_loadings_combined_tui",
                     "efa_communalities_combined_tui",
                     "efa_explained_variance_combined_tui"):
            src = PCA_DIR / f"{stem}.csv"
            dst = PCA_DIR / f"{stem}_{variant}.csv"
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  Saved → {dst.name}")
            else:
                print(f"  WARNING: {src.name} not found, skipping copy")

        # 2-4. All regressions in parallel
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

        # 5. Export tables
        run([PYTHON, REG_DIR / "export_reg_tables_efa_main.py"], "export EFA tables", skip)

        # 6. Copy Excel output to variant-specific name
        src = REG_DIR / "reg_tables_efa_main.xlsx"
        dst = REG_DIR / f"reg_tables_efa_main_{variant}.xlsx"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Saved → {dst.name}")
        else:
            print(f"  WARNING: {src.name} not found, skipping copy")

    print(f"\n{'#' * 70}")
    print("  ALL EFA VARIANTS COMPLETE")
    print(f"{'#' * 70}")
    for variant, _ in variants:
        p = REG_DIR / f"reg_tables_efa_main_{variant}.xlsx"
        mark = "OK" if p.exists() else "MISSING"
        print(f"  [{mark}] {p.name}")


if __name__ == "__main__":
    main()
