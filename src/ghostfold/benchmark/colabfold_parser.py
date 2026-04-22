import json
from pathlib import Path


def parse_best_scores(colabfold_out_dir: Path) -> dict:
    """Parse the rank_001 JSON and locate the best PDB from a ColabFold output dir.

    ColabFold writes per-model score files named:
        {query}_scores_rank_001_alphafold2_ptm_model_N_seed_XXX.json
    and the corresponding structure:
        {query}_unrelaxed_rank_001_*.pdb  (or _relaxed_rank_001_*.pdb)

    Returns a dict with keys: ptm, mean_plddt, max_pae (or None), best_pdb (Path).

    Raises FileNotFoundError if no rank_001 score file is found.
    """
    score_files = sorted(colabfold_out_dir.glob("*_scores_rank_001*.json"))
    if not score_files:
        raise FileNotFoundError(
            f"No rank_001 score JSON found in {colabfold_out_dir}"
        )
    score_file = score_files[0]

    with open(score_file) as fh:
        data = json.load(fh)

    plddt_vals = data.get("plddt", [])
    mean_plddt = float(sum(plddt_vals) / len(plddt_vals)) if plddt_vals else None

    result: dict = {
        "ptm": data.get("ptm"),
        "mean_plddt": mean_plddt,
        "max_pae": data.get("max_pae"),
        "best_pdb": None,
    }

    # Prefer relaxed structure; fall back to unrelaxed
    for pattern in ("*_relaxed_rank_001*.pdb", "*_unrelaxed_rank_001*.pdb"):
        pdb_files = sorted(colabfold_out_dir.glob(pattern))
        if pdb_files:
            result["best_pdb"] = pdb_files[0]
            break

    return result
