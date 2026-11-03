from functools import lru_cache
from itertools import combinations

from rdkit import Chem
from rdkit.Chem import FunctionalGroups


@lru_cache(maxsize=1)
def load_rdkit_functional_group_patterns():
    roots = FunctionalGroups.BuildFuncGroupHierarchy()
    patterns = []

    def visit(node):
        pattern = getattr(node, "pattern", None)
        if pattern is not None:
            name = getattr(node, "label", None) or getattr(node, "name", None)
            if name is None:
                name = "functional_group"
            patterns.append(
                {
                    "name": str(name),
                    "smarts": getattr(node, "smarts", ""),
                    "pattern": pattern,
                }
            )
        for child in getattr(node, "children", []):
            visit(child)

    for root in roots:
        visit(root)

    return tuple(patterns)


def identify_functional_groups(smiles, mol_id, atom_offset=0):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    groups = []
    try:
        patterns = load_rdkit_functional_group_patterns()
    except Exception:
        return []

    for fg_def in patterns:
        try:
            matches = mol.GetSubstructMatches(fg_def["pattern"], uniquify=True)
        except Exception:
            continue

        for match_id, match in enumerate(matches):
            local_atoms = tuple(sorted(set(int(atom) for atom in match)))
            if not local_atoms:
                continue
            global_atoms = tuple(atom_offset + atom for atom in local_atoms)
            groups.append(
                {
                    "mol_id": int(mol_id),
                    "name": fg_def["name"],
                    "smarts": fg_def["smarts"],
                    "match_id": int(match_id),
                    "local_atoms": local_atoms,
                    "atoms": global_atoms,
                }
            )

    return groups


def build_functional_group_lookup(functional_groups, num_nodes):
    atom_to_groups = [[] for _ in range(num_nodes)]
    groups = []

    for group in functional_groups or []:
        atoms = tuple(sorted(set(int(atom) for atom in group.get("atoms", []))))
        atoms = tuple(atom for atom in atoms if 0 <= atom < num_nodes)
        if not atoms:
            continue

        group_id = len(groups)
        groups.append(
            {
                "mol_id": group.get("mol_id"),
                "name": str(group.get("name", "functional_group")),
                "smarts": group.get("smarts", ""),
                "match_id": group.get("match_id"),
                "local_atoms": tuple(int(atom) for atom in group.get("local_atoms", ())),
                "atoms": atoms,
            }
        )
        for atom in atoms:
            atom_to_groups[atom].append(group_id)

    return {
        "groups": groups,
        "atom_to_groups": atom_to_groups,
    }


def score_functional_group_path(path, lookup=None, return_annotation=False):
    path = tuple(int(atom) for atom in path)
    path_atoms = set(path)
    empty_annotation = {
        "path": list(path),
        "contains_fg": False,
        "cover_full_fg": False,
        "is_fg_association": False,
        "is_fg_type_association": False,
        "hit_fgs": [],
        "cover_full_fgs": [],
        "association_pairs": [],
        "hit_fg_count": 0,
        "fg_score": 0.0,
    }

    if lookup is None or not path_atoms:
        if return_annotation:
            return 0.0, empty_annotation
        return 0.0

    atom_to_groups = lookup.get("atom_to_groups", [])
    groups = lookup.get("groups", [])
    hit_group_ids = set()
    for atom in path_atoms:
        if 0 <= atom < len(atom_to_groups):
            hit_group_ids.update(atom_to_groups[atom])

    if not hit_group_ids:
        if return_annotation:
            return 0.0, empty_annotation
        return 0.0

    hit_names = []
    cover_full_names = []
    overlap_score = 0.0
    hit_instance_atoms = set()

    for group_id in sorted(hit_group_ids):
        group = groups[group_id]
        group_atoms = set(group["atoms"])
        overlap = path_atoms & group_atoms
        if not overlap:
            continue
        instance_key = tuple(sorted(group_atoms))
        hit_names.append(group["name"])
        if instance_key not in hit_instance_atoms:
            overlap_score += len(overlap) / float(len(group_atoms))
            hit_instance_atoms.add(instance_key)
        if group_atoms.issubset(path_atoms):
            cover_full_names.append(group["name"])

    unique_hit_names = sorted(set(hit_names))
    unique_full_names = sorted(set(cover_full_names))
    is_type_association = len(hit_instance_atoms) >= 2 and len(unique_hit_names) >= 2
    if is_type_association:
        association_pairs = [
            "{}|{}".format(left, right)
            for left, right in combinations(unique_hit_names, 2)
        ]
    else:
        association_pairs = []

    score = 0.5
    score += min(overlap_score, 2.0)
    if unique_full_names:
        score += min(len(unique_full_names), 3)
    if len(hit_instance_atoms) >= 2:
        score += 1.5
    if is_type_association:
        score += 2.0
    score = min(score, 8.0)

    if not return_annotation:
        return score

    annotation = {
        "path": list(path),
        "contains_fg": True,
        "cover_full_fg": bool(unique_full_names),
        "is_fg_association": len(hit_instance_atoms) >= 2,
        "is_fg_type_association": is_type_association,
        "hit_fgs": unique_hit_names,
        "cover_full_fgs": unique_full_names,
        "association_pairs": association_pairs,
        "hit_fg_count": int(len(hit_instance_atoms)),
        "fg_score": float(score),
    }
    return score, annotation
