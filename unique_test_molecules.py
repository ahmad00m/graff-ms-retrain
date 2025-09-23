import json
from pathlib import Path
from typing import Optional, Dict, Set, List

import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import inchi as rd_inchi
from modifinder.classes.Compound import Compound

files = [
    "BERKELEY-LAB.csv",
    "GNPS-MSMLS.csv",
    "GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_POSITIVE.csv",
    "GNPS-NIH-SMALLMOLECULEPHARMACOLOGICALLYACTIVE.csv",
]
Path("data/nist-20/test_set").mkdir(parents=True, exist_ok=True)
OUT_JSON = "data/nist-20/test_set/exclude_inchikey2d.json"

ID_COLS = ["id_bigger", "id_smaller"]

################################################################
# Helpers
################################################################

def resolve_mol_with_modifinder(molecule_id: str) -> Optional[Chem.Mol]:
    """
    Build a ModiFinder Compound and return its RDKit Mol (compound.structure).
    """
    try:
        c = Compound(molecule_id)
        if getattr(c, "structure", None) is not None:
            return c.structure
    except Exception:
        return None
    return None

def mol_to_smiles(mol: Optional[Chem.Mol]) -> str:
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""

def mol_to_inchikey(mol: Optional[Chem.Mol]) -> Optional[str]:
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        pass
    if rd_inchi is not None:
        try:
            return rd_inchi.MolToInchiKey(mol)
        except Exception:
            return None
    return None

def ik2d(ikey: Optional[str]) -> Optional[str]:
    if not isinstance(ikey, str):
        return None
    parts = ikey.split("-")
    return parts[0] if parts and parts[0] else None

################################################################
# Collect unique IDs
################################################################

id_to_source: Dict[str, str] = {}
unique_ids: Set[str] = set()

for f in files:
    p = Path("data/nist-20/library") / f
    if not p.exists():
        print(f"[WARN] Missing file: {p}")
        continue
    df = pd.read_csv(p, dtype=str, low_memory=False)
    missing = [c for c in ID_COLS if c not in df.columns]
    if missing:
        print(f"[WARN] {p.name} missing expected columns: {missing}. Present: {list(df.columns)[:10]}...")
    for col in [c for c in ID_COLS if c in df.columns]:
        for mid in df[col].dropna().astype(str).str.strip().unique():
            if not mid:
                continue
            if mid not in unique_ids:
                unique_ids.add(mid)
                id_to_source[mid] = p.stem

print(f"[INFO] Collected {len(unique_ids)} unique IDs from {len(files)} files.")

################################################################
# Resolve via ModiFinder → Mol → SMILES/InChIKey/InChIKey2D
################################################################

details: List[Dict] = []
exclude2d: Set[str] = set()

for mid in tqdm(sorted(unique_ids), desc="Resolving via ModiFinder"):
    mol = resolve_mol_with_modifinder(mid)
    smiles = mol_to_smiles(mol)
    ikey = mol_to_inchikey(mol)
    two_d = ik2d(ikey)

    status = (
        "resolved" if two_d else
        ("smiles_ok_but_inchikey_failed" if smiles else "unresolved")
    )

    if two_d:
        exclude2d.add(two_d)

    details.append({
        "source": id_to_source.get(mid, ""),
        "molecule_id": mid,
        "smiles": smiles,
        "inchikey": ikey or "",
        "inchikey2d": two_d or "",
        "status": status,
    })

################################################################
# Write outputs
################################################################

with open(OUT_JSON, "w") as f:
    json.dump({"exclude_inchikey2d": sorted(exclude2d)}, f, indent=2)

print(f"[DONE] Wrote {len(exclude2d)} unique InChIKey2D to {OUT_JSON}")