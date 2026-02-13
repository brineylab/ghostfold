from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path


def postprocess_msa_outputs(project_name: str) -> None:
    """Post-process MSA outputs: fix FASTA headers and create A3M copies.

    Finds all pstMSA.fasta files under the project, rewrites the first header
    to match the parent directory name, and creates a .a3m copy.

    Args:
        project_name: Path to the project directory.
    """
    pattern = os.path.join(project_name, "**", "pstMSA.fasta")
    fasta_files = glob.glob(pattern, recursive=True)

    for fasta_path in fasta_files:
        header_id = os.path.basename(os.path.dirname(fasta_path))

        # Read lines, rewrite first header
        with open(fasta_path, "r") as f:
            lines = f.readlines()

        if lines and lines[0].startswith(">"):
            lines[0] = f">{header_id}\n"

        with open(fasta_path, "w") as f:
            f.writelines(lines)

        # Create .a3m copy
        a3m_path = fasta_path.replace(".fasta", ".a3m")
        shutil.copy2(fasta_path, a3m_path)
        print(f"   [Processed MSA] {header_id}")


def cleanup_colabfold_outputs(subsample_dir: str) -> None:
    """Organize ColabFold output files into subdirectories.

    For each prediction directory under subsample_dir/preds/:
    - Moves .json files to scores/
    - Moves .png files to imgs/
    - Moves recycle PDB files (.r0.pdb, .r1.pdb, etc.) to recycles/
    - Copies the rank_001 PDB to best/
    - Deletes done.txt files

    Args:
        subsample_dir: Path to the subsample directory.
    """
    preds_dir = os.path.join(subsample_dir, "preds")
    best_dir = os.path.join(subsample_dir, "best")

    print(f"---\nStarting cleanup for: {subsample_dir}")

    os.makedirs(best_dir, exist_ok=True)

    if not os.path.isdir(preds_dir):
        print(f"No preds directory found at {preds_dir}, skipping cleanup.")
        return

    for pred_name in os.listdir(preds_dir):
        pred_dir = os.path.join(preds_dir, pred_name)
        if not os.path.isdir(pred_dir):
            continue

        print(f"   [Cleaning] {pred_name}")

        # Create subdirectories
        scores_dir = os.path.join(pred_dir, "scores")
        imgs_dir = os.path.join(pred_dir, "imgs")
        recycles_dir = os.path.join(pred_dir, "recycles")
        os.makedirs(scores_dir, exist_ok=True)
        os.makedirs(imgs_dir, exist_ok=True)
        os.makedirs(recycles_dir, exist_ok=True)

        rank_1_pdb = None

        for fname in os.listdir(pred_dir):
            fpath = os.path.join(pred_dir, fname)
            if not os.path.isfile(fpath):
                continue

            # Move JSON files to scores/
            if fname.endswith(".json"):
                shutil.move(fpath, os.path.join(scores_dir, fname))
            # Move PNG files to imgs/
            elif fname.endswith(".png"):
                shutil.move(fpath, os.path.join(imgs_dir, fname))
            # Move recycle PDB files to recycles/
            elif _is_recycle_pdb(fname):
                shutil.move(fpath, os.path.join(recycles_dir, fname))
            # Track rank_001 PDB
            elif "rank_001" in fname and fname.endswith(".pdb"):
                rank_1_pdb = fpath
            # Delete done.txt files
            elif fname.endswith("done.txt"):
                os.remove(fpath)

        # Copy top-ranked PDB to best/
        if rank_1_pdb and os.path.exists(rank_1_pdb):
            dest_pdb = os.path.join(best_dir, f"{pred_name}_ghostfold.pdb")
            shutil.copy2(rank_1_pdb, dest_pdb)
            print(f"     -> Copied top PDB to best/{os.path.basename(dest_pdb)}")

    print(f"Cleanup complete for {subsample_dir}.")


def _is_recycle_pdb(fname: str) -> bool:
    """Check if a filename is a recycle PDB (e.g., *.r0.pdb, *.r12.pdb)."""
    if not fname.endswith(".pdb"):
        return False
    # Match patterns like .r0.pdb, .r1.pdb, .r10.pdb
    parts = fname.rsplit(".", 2)
    if len(parts) >= 3:
        middle = parts[-2]
        return middle.startswith("r") and middle[1:].isdigit()
    return False
