#!/usr/bin/env python3
"""Validate the transferred production dataset and record server provenance.

Run with the experiment environment's Python. No accelerator is needed. Source
hash differences are recorded because server changes are allowed; the original
processed manifest must retain its handoff hash.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys

import numpy as np
import yaml


AA_NAMES = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLY", "GLU", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
AA_CODES = "ARNDCQGEHILKMFPSTWYV"
AA_INDEX = {code: index for index, code in enumerate(AA_CODES)}
EXPECTED_SPLITS = {"train": 175, "val": 75, "test": 92}
PAPER_PEPTIDES = {"AC", "AP", "AT", "ET", "HP", "HT", "IM", "KQ", "KS", "LW", "NF", "NY", "RL", "RV", "TD"}


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def expected_features(sequence):
    return np.asarray([[AA_INDEX[sequence[residue]], atom, element, residue]
                       for residue in range(2)
                       for atom, element in enumerate((1, 0, 0, 0, 2))], dtype=np.int64)


def validate(root, report):
    errors = report["errors"]

    def check(condition, description):
        if not condition:
            errors.append(description)
        return condition

    config = yaml.safe_load((root / "configs/dipeptides.yaml").read_text())
    configured_data = Path(config["data"]["root"])
    data_root = configured_data if configured_data.is_absolute() else root / configured_data
    cache = data_root / "processed"
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    handoff = json.loads((data_root / "results/handoff-checks.json").read_text())
    check(manifest.get("version") == 1, "Processed manifest version is not 1")
    check(manifest.get("units") == "nm", "Processed coordinate units are not nm")
    check(set(manifest["splits"]) == set(EXPECTED_SPLITS), "Manifest split names differ from train/val/test")

    # The tar listing includes directories. Check every listed data member, even
    # reports and auxiliary feature files that training does not open directly.
    member_list = data_root / "handoff_archive_members.txt"
    member_files, member_dirs = [], []
    for entry in member_list.read_text().splitlines():
        member = PurePosixPath(entry)
        if not check(not member.is_absolute() and ".." not in member.parts,
                     f"Unsafe archive member: {entry}"):
            continue
        if member.parts and member.parts[0] == "ggf_md":
            member = PurePosixPath(*member.parts[1:])
        if not member.parts or member.parts[0] != "data":
            continue
        path = root.joinpath(*member.parts)
        if entry.endswith("/"):
            member_dirs.append(str(member))
            check(path.is_dir(), f"Missing extracted archive directory: {member}")
        else:
            member_files.append(str(member))
            check(path.is_file(), f"Missing extracted archive file: {member}")
    check(bool(member_files), "Archive listing contains no data files")
    report["archive_members"] = {"listing": str(member_list.relative_to(root)),
                                 "listing_sha256": sha256(member_list),
                                 "data_files": len(member_files),
                                 "data_directories": len(member_dirs)}

    baseline_hashes = {}
    for relative, expected in handoff["files_sha256"].items():
        path = root / relative
        actual = sha256(path) if path.is_file() else None
        baseline_hashes[relative] = {"handoff_sha256": expected, "current_sha256": actual,
                                     "matches_handoff": actual == expected}
        check(actual is not None, f"Missing original handoff source file: {relative}")
        if relative == "data/processed/manifest.json":
            check(actual == expected, "Processed manifest hash differs from the verified handoff")
    report["handoff_hash_comparison"] = baseline_hashes
    report["modified_source_files"] = [path for path, item in baseline_hashes.items()
                                       if not item["matches_handoff"]]

    sequence_sets = {split: set(items) for split, items in manifest["splits"].items()}
    for split, count in EXPECTED_SPLITS.items():
        check(len(sequence_sets.get(split, ())) == count,
              f"{split}: expected {count} peptide identities")
    for index, left in enumerate(EXPECTED_SPLITS):
        for right in list(EXPECTED_SPLITS)[index + 1:]:
            overlap = sorted(sequence_sets.get(left, set()) & sequence_sets.get(right, set()))
            check(not overlap, f"Identity leakage across {left}/{right}: {overlap}")

    selected = config["evaluation"]["peptides"]
    check(len(selected) == 15 and len(set(selected)) == 15 and set(selected) == PAPER_PEPTIDES,
          "Evaluation peptide list differs from the paper's 15 held-out identities")
    check(set(selected) <= sequence_sets.get("test", set()),
          "Selected evaluation peptides are missing from the test split")
    report["evaluation_peptides"] = selected

    moments = np.zeros(3, dtype=np.float64)
    summaries, arrays = {}, {}
    processed_bytes = manifest_path.stat().st_size
    expected_array_paths = set()
    for split, items in manifest["splits"].items():
        total_frames = 0
        for sequence, info in items.items():
            if not check(len(sequence) == 2 and set(sequence) <= set(AA_CODES) and "G" not in sequence,
                         f"Invalid nonglycine dipeptide identity: {split}/{sequence}"):
                continue
            coords_path = cache / split / f"{sequence}.npy"
            features_path = cache / split / f"{sequence}.features.npy"
            expected_array_paths.update((coords_path, features_path))
            if not check(coords_path.is_file() and features_path.is_file(),
                         f"Missing coordinate/feature array: {split}/{sequence}"):
                continue
            x = np.load(coords_path, mmap_mode="r", allow_pickle=False)
            features = np.load(features_path, allow_pickle=False)
            check(x.shape == (info["samples"], 30), f"Coordinate shape mismatch: {split}/{sequence}")
            check(x.dtype == np.float32, f"Coordinates are not float32: {split}/{sequence}")
            check(features.shape == (10, 4) and np.issubdtype(features.dtype, np.integer),
                  f"Feature shape/dtype mismatch: {split}/{sequence}")
            check(np.array_equal(features, np.asarray(info["features"])),
                  f"Feature file differs from manifest: {split}/{sequence}")
            check(np.array_equal(features, expected_features(sequence)),
                  f"Canonical atom, element, residue or sequence features differ: {split}/{sequence}")
            for begin in range(0, len(x), 8192):
                block = np.asarray(x[begin:begin + 8192], dtype=np.float64)
                check(np.isfinite(block).all(), f"Nonfinite coordinates: {split}/{sequence} at frame {begin}")
                if split == "train":
                    moments += (block.size, block.sum(), np.square(block).sum())
            total_frames += len(x)
            for path in (coords_path, features_path):
                size = path.stat().st_size
                processed_bytes += size
                arrays[str(path.relative_to(root))] = {"bytes": size, "sha256": sha256(path)}
        summaries[split] = {"peptides": len(items), "frames": total_frames,
                            "coordinate_dimensions": 30}
    actual_arrays = set(cache.glob("*/*.npy"))
    check(actual_arrays == expected_array_paths,
          f"Unexpected/missing processed arrays: {sorted(str(p.relative_to(root)) for p in actual_arrays ^ expected_array_paths)}")
    check(processed_bytes == handoff["processed_data_bytes"],
          f"Processed dataset byte count differs: {processed_bytes} != {handoff['processed_data_bytes']}")
    variance = moments[2] / moments[0] - (moments[1] / moments[0]) ** 2
    norm = float(1 / np.sqrt(variance))
    check(np.isfinite(norm) and np.isclose(norm, manifest["norm_factor"], rtol=1e-12, atol=0),
          f"Train-only normalization differs: {norm} != {manifest['norm_factor']}")
    report["dataset"] = {"splits": summaries, "processed_bytes": processed_bytes,
                          "array_files": len(arrays), "finite_coordinates_checked": True,
                          "canonical_features_checked": True, "identity_overlap_checked": True,
                          "normalization_source": "training coordinates only",
                          "normalization_scalar_count": int(moments[0]),
                          "recomputed_norm_factor": norm,
                          "manifest_norm_factor": manifest["norm_factor"], "arrays": arrays}
    frames = summaries["train"]["frames"]
    steps_per_epoch = frames // config["training"]["batch_size"]
    report["required_training"] = {
        "steps_per_epoch": steps_per_epoch,
        "diffusion": {"epochs": config["training"]["epochs"],
                      "optimizer_steps": steps_per_epoch * config["training"]["epochs"]},
        "fokker_planck": {"epochs": config["training"]["epochs"],
                         "optimizer_steps": steps_per_epoch * config["training"]["epochs"]},
        "mixture": {"epochs": sum(config["training"]["mixture_epochs"]),
                    "stage_epochs": config["training"]["mixture_epochs"],
                    "optimizer_steps": steps_per_epoch * sum(config["training"]["mixture_epochs"])}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Default: data/results/server-validation.json")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "data/results/server-validation.json"
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "root": str(root),
              "validator_sha256": sha256(Path(__file__)), "python": sys.version,
              "numpy": np.__version__, "passed": False, "errors": []}
    try:
        validate(root, report)
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    report["passed"] = not report["errors"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, output)
    print(json.dumps({key: value for key, value in report.items()
                      if key in ("passed", "errors", "modified_source_files", "required_training")}, indent=2))
    print(f"Validation report: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
