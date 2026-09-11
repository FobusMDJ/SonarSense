"""Rebuilds train/val/test as a genuine, leakage-safe 3-way split of the
whole preprocessed YOLO dataset, using MANIFEST.csv's orig_image field to
group tiles by their real-world source (wreck site, recording session,
survey pass) rather than by tile identity -- so no two tiles cut from the
same underlying acquisition end up in different splits.

WHY THIS EXISTS: data.yaml currently has val==test (no genuine held-out
val), and an audit of MANIFEST.csv against site-level grouping found real
train/test leakage under the CURRENT split too, not just the val==test
problem:
    ai4shipwrecks:        0.0% of tiles leaked (already effectively site-safe)
    subpipe_mini_sss:    99.7% of tiles leaked -- this source is ONE
                          continuous ~1.16-hour recording (timestamps span
                          4173s with a median 0s gap between consecutive
                          tiles), so a random per-tile train/test split
                          scatters near-duplicate frames across both --
                          this is exactly the "same continuous acquisition
                          sequence" leakage the project's own instructions
                          warn about, and it affects `pipe`, currently the
                          best-performing class (AP50 0.835) -- that number
                          may be inflated by frame memorization, not
                          genuine generalization.
    aquascan_1k:         61.5% leaked (grouped by capture date)
    uatd_cylinders:       0.6% leaked (Roboflow near-duplicate frames)
    crabpot_ghostnet_proxy (net_test): 1.8% leaked (grouped by Rec-N tow pass)

SITE-KEY RULES (per source, from actual filename/timestamp structure --
inspect build_site_key() before trusting a new source blindly):
  ai4shipwrecks         -> wreck name (orig_image stem with the trailing
                            _NN segment index stripped) -- discrete sites,
                            group-shuffled.
  aquascan_1k           -> capture date (from the Screenshot_YYYY-MM-DD_
                            filename) -- discrete sessions, group-shuffled.
  uatd_cylinders        -> leading frame-number prefix before "_bmp.rf." --
                            discrete-ish, group-shuffled.
  crabpot_ghostnet_proxy -> "RecN" tow-pass id -- discrete passes,
                            group-shuffled.
  subpipe_mini_sss      -> NOT grouped by key at all -- this source is
                            effectively one continuous recording, so
                            group-shuffling would either put almost
                            everything in one split or require an
                            arbitrarily-chosen correlation-time constant.
                            Instead: sorted by timestamp (hf/lf bands
                            handled separately) and cut into CONTIGUOUS
                            time blocks (first ~80% by duration -> train,
                            next ~10% -> val, last ~10% -> test), which is
                            the standard leakage-safe approach for a single
                            continuous sensor recording. Tiles sharing the
                            exact same timestamp (hf/lf pairs at one
                            instant) are kept as one atomic unit.

This does NOT retrain or re-evaluate anything -- it only rebuilds the
split and copies files (originals untouched). best.pt was trained on the
OLD train/ folder; Phase-1's error_analysis.py numbers were measured
against the OLD test/ folder and remain valid AS a report of best.pt
against that (leaky) set -- they are not automatically comparable to
anything measured against the new split. Retrain on train_v2/ and re-run
error_analysis.py against test_v2/ to get a trustworthy corrected
baseline before trusting any of Experiments B-G in the roadmap.

Usage:
    python -m src.preprocessing.site_aware_split \\
        --manifest data/processed/yolo_dataset_v2_preprocessed/MANIFEST.csv \\
        --dataset_root data/processed/yolo_dataset_v2_preprocessed \\
        --out_root data/processed/yolo_dataset_v2_preprocessed \\
        --train_frac 0.8 --val_frac 0.1 --test_frac 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path

from src.utils.config import get_logger

logger = get_logger(__name__)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def build_site_key(row: dict) -> tuple[str, float | None]:
    """Returns (site_key, timestamp_or_None). timestamp is only set for the
    continuous-recording source (subpipe) -- everything else is grouped by
    site_key alone and timestamp is irrelevant."""
    src = row["source"]
    orig = row["orig_image"]
    if src == "ai4shipwrecks":
        stem = orig.split("/")[-1].rsplit(".", 1)[0]
        m = re.match(r"^(.*)_(\d+)$", stem)
        return f"ai4sw:{m.group(1) if m else stem}", None
    if src == "subpipe_mini_sss":
        stem = orig.split("/")[-1].rsplit(".", 1)[0]
        band = "hf" if "HF" in orig else ("lf" if "LF" in orig else "?")
        try:
            ts = float(stem)
        except ValueError:
            ts = None
        return f"subpipe:{band}", ts
    if src == "aquascan_1k":
        m = re.search(r"Screenshot_(\d{4}-\d{2}-\d{2})_", orig)
        return f"aquascan:{m.group(1) if m else orig}", None
    if src == "uatd_cylinders":
        stem = orig.split("/")[-1]
        m = re.match(r"^(\d+)_", stem)
        return f"cyl:{m.group(1) if m else stem}", None
    if src == "crabpot_ghostnet_proxy":
        stem = orig.split("/")[-1]
        m = re.match(r"^(Rec\d+)_", stem)
        return f"net:{m.group(1) if m else stem}", None
    return f"other:{orig}", None


def audit_current_leakage(rows: list[dict]) -> None:
    key_to_splits = defaultdict(set)
    for r in rows:
        key_to_splits[r["site_key"]].add(r["split"])
    by_source = defaultdict(lambda: [0, 0])
    for r in rows:
        by_source[r["source"]][0] += 1
        if len(key_to_splits[r["site_key"]]) > 1:
            by_source[r["source"]][1] += 1
    logger.info("=== Leakage audit of the CURRENT train/test split (before rebuilding) ===")
    for src, (tot, leaked) in by_source.items():
        logger.info("  %s: %d/%d tiles (%.1f%%) share a site with a tile in the other split",
                    src, leaked, tot, 100 * leaked / tot if tot else 0.0)


def split_continuous(rows: list[dict], train_frac: float, val_frac: float) -> dict[str, str]:
    """Contiguous time-block split for the subpipe rows. Returns {out_stem: split_name}."""
    with_ts = [r for r in rows if r["_ts"] is not None]
    without_ts = [r for r in rows if r["_ts"] is None]
    if without_ts:
        logger.warning("%d subpipe rows had no parseable timestamp -- assigned to train", len(without_ts))

    with_ts.sort(key=lambda r: r["_ts"])
    unique_ts = sorted(set(r["_ts"] for r in with_ts))
    n = len(unique_ts)
    train_cut = unique_ts[int(n * train_frac)] if n else 0
    val_cut = unique_ts[int(n * (train_frac + val_frac))] if n else 0

    assignment = {}
    for r in with_ts:
        if r["_ts"] <= train_cut:
            split = "train"
        elif r["_ts"] <= val_cut:
            split = "val"
        else:
            split = "test"
        assignment[r["out_stem"]] = split
    for r in without_ts:
        assignment[r["out_stem"]] = "train"
    return assignment


def split_grouped(rows: list[dict], train_frac: float, val_frac: float, seed: int) -> dict[str, str]:
    """Group-shuffled split for discrete-site sources: every row sharing a
    site_key gets the same split. Greedy streaming balance keeps split
    sizes close to the target fractions without needing per-class
    stratification math -- documented as best-effort, not exact."""
    import random

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["site_key"]].append(r)

    keys = list(groups.keys())
    random.Random(seed).shuffle(keys)

    target = {"train": train_frac, "val": val_frac, "test": 1 - train_frac - val_frac}
    counts = {"train": 0, "val": 0, "test": 0}
    assignment = {}
    for key in keys:
        group_rows = groups[key]
        # assign to whichever split is furthest below its target share
        total_assigned = sum(counts.values()) or 1
        deficits = {s: target[s] - counts[s] / total_assigned for s in target}
        chosen = max(deficits, key=deficits.get)
        for r in group_rows:
            assignment[r["out_stem"]] = chosen
        counts[chosen] += len(group_rows)
    return assignment


def find_source_file(dataset_root: Path, old_split: str, out_stem: str, subfolder: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES if subfolder == "images" else (".txt",):
        candidate = dataset_root / old_split / subfolder / f"{out_stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)

    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    logger.info("Loaded %d rows from %s", len(rows), manifest_path)

    for r in rows:
        key, ts = build_site_key(r)
        r["site_key"] = key
        r["_ts"] = ts

    audit_current_leakage(rows)

    subpipe_rows = [r for r in rows if r["source"] == "subpipe_mini_sss"]
    other_rows = [r for r in rows if r["source"] != "subpipe_mini_sss"]

    assignment: dict[str, str] = {}
    if subpipe_rows:
        assignment.update(split_continuous(subpipe_rows, args.train_frac, args.val_frac))
    if other_rows:
        assignment.update(split_grouped(other_rows, args.train_frac, args.val_frac, args.seed))

    # Re-audit the NEW assignment to prove it's actually leakage-free.
    for r in rows:
        r["new_split"] = assignment.get(r["out_stem"], "train")
    key_to_new_splits = defaultdict(set)
    for r in rows:
        if r["source"] != "subpipe_mini_sss":  # continuous source has no discrete key to check
            key_to_new_splits[r["site_key"]].add(r["new_split"])
    still_leaked = sum(1 for k, v in key_to_new_splits.items() if len(v) > 1)
    logger.info("New split: %d/%d discrete site_keys still span multiple splits (should be 0)",
                still_leaked, len(key_to_new_splits))

    counts = defaultdict(int)
    for r in rows:
        counts[r["new_split"]] += 1
    logger.info("New split sizes: train=%d val=%d test=%d", counts["train"], counts["val"], counts["test"])

    # Copy files into train_v2/val_v2/test_v2.
    manifest_out_rows = []
    n_copied, n_missing = 0, 0
    for r in rows:
        out_stem = r["out_stem"]
        new_split = r["new_split"]
        old_split = r["split"]
        dst_images = out_root / f"{new_split}_v2" / "images"
        dst_labels = out_root / f"{new_split}_v2" / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        img_src = find_source_file(dataset_root, old_split, out_stem, "images")
        if img_src is None:
            n_missing += 1
            logger.warning("Missing source image for %s (expected under %s/images)", out_stem, old_split)
            continue
        shutil.copy2(img_src, dst_images / img_src.name)

        lbl_src = find_source_file(dataset_root, old_split, out_stem, "labels")
        if lbl_src is not None:
            shutil.copy2(lbl_src, dst_labels / lbl_src.name)

        n_copied += 1
        manifest_out_rows.append({
            "out_stem": out_stem, "source": r["source"], "site_key": r["site_key"],
            "old_split": old_split, "new_split": new_split,
        })

    manifest_out_path = out_root / "split_manifest_v2.csv"
    with open(manifest_out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["out_stem", "source", "site_key", "old_split", "new_split"])
        writer.writeheader()
        writer.writerows(manifest_out_rows)

    data_yaml_path = out_root / "data_v2.yaml"
    with open(args.data_yaml) as f:
        import yaml
        base_cfg = yaml.safe_load(f)
    new_cfg = {
        "train": "train_v2/images", "val": "val_v2/images", "test": "test_v2/images",
        "nc": base_cfg["nc"], "names": base_cfg["names"],
    }
    with open(data_yaml_path, "w") as f:
        import yaml
        yaml.safe_dump(new_cfg, f, sort_keys=False)

    logger.info("Copied %d tiles (%d missing sources) -> %s/{train_v2,val_v2,test_v2}", n_copied, n_missing, out_root)
    logger.info("Wrote %s and %s", manifest_out_path, data_yaml_path)
    logger.info("NEXT STEP: retrain on train_v2 and re-run error_analysis.py against test_v2 "
                "for a corrected, leakage-safe baseline before trusting Experiments B-G.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--dataset_root", type=str, required=True, help="Folder containing the CURRENT train/ and test/ subfolders.")
    p.add_argument("--out_root", type=str, required=True, help="Where train_v2/val_v2/test_v2 + data_v2.yaml are written.")
    p.add_argument("--data_yaml", type=str, default=None, help="Existing data.yaml to copy nc/names from (default: <dataset_root>/data.yaml).")
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1, help="Informational only -- actual test share is 1 - train_frac - val_frac.")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.data_yaml is None:
        args.data_yaml = str(Path(args.dataset_root) / "data.yaml")
    if abs(args.train_frac + args.val_frac + args.test_frac - 1.0) > 1e-6:
        raise SystemExit("--train_frac + --val_frac + --test_frac must sum to 1.0")
    run(args)
