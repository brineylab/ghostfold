from pathlib import Path

import numpy as np

try:
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb_io

    _BIOTITE_AVAILABLE = True
except ImportError:
    _BIOTITE_AVAILABLE = False


def _load_ca_coords(pdb_path: Path) -> np.ndarray:
    """Load Cα coordinates from a PDB file as an (N, 3) float array."""
    if not _BIOTITE_AVAILABLE:
        raise ImportError(
            "biotite is required for structure metrics. "
            "Install with: pip install 'ghostfold[bench]'"
        )
    pdb_file = pdb_io.PDBFile.read(str(pdb_path))
    structure = pdb_file.get_structure(model=1)
    aa_mask = struc.filter_amino_acids(structure)
    ca_mask = structure.atom_name == "CA"
    ca_atoms = structure[aa_mask & ca_mask]
    return ca_atoms.coord.astype(np.float64)  # (L, 3)


def _kabsch_rmsd(ref: np.ndarray, mobile: np.ndarray) -> tuple[np.ndarray, float]:
    """Kabsch superimposition; returns (superimposed mobile coords, RMSD)."""
    ref_c = ref - ref.mean(axis=0)
    mob_c = mobile - mobile.mean(axis=0)

    H = mob_c.T @ ref_c
    U, _, Vt = np.linalg.svd(H)
    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    superimposed = mob_c @ R.T + ref.mean(axis=0)

    rmsd = float(np.sqrt(np.mean(np.sum((ref - superimposed) ** 2, axis=1))))
    return superimposed, rmsd


def _tm_score(ref_coords: np.ndarray, aligned_coords: np.ndarray, l_ref: int) -> float:
    """TM-score of aligned_coords relative to ref_coords with reference length l_ref."""
    if l_ref < 22:
        d0 = 0.5
    else:
        d0 = 1.24 * (l_ref - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)
    d_sq = np.sum((ref_coords - aligned_coords) ** 2, axis=1)
    return float(np.sum(1.0 / (1.0 + d_sq / d0 ** 2)) / l_ref)


def compute_structure_metrics(predicted_pdb: Path, reference_pdb: Path) -> dict:
    """Superimpose predicted onto reference and return RMSD + TM-score.

    Both PDBs are trimmed to their common Cα count before alignment.
    TM-score is normalised by the *reference* structure length, matching
    the convention used by TM-align and the original Zhang & Skolnick paper.
    """
    pred_coords = _load_ca_coords(predicted_pdb)
    ref_coords = _load_ca_coords(reference_pdb)

    l_ref = len(ref_coords)
    min_len = min(len(pred_coords), l_ref)
    ref_trimmed = ref_coords[:min_len]
    pred_trimmed = pred_coords[:min_len]

    superimposed, rmsd = _kabsch_rmsd(ref_trimmed, pred_trimmed)
    tm = _tm_score(ref_trimmed, superimposed, l_ref)

    return {"rmsd": rmsd, "tm_score": tm}
