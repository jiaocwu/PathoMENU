
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa


ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
STANDARD_AMINO_ACIDS = set(ONE_TO_THREE.values())
METAL_ELEMENTS = {
    "LI", "BE", "NA", "MG", "K", "CA", "V", "CR", "MN", "FE", "CO", "NI",
    "CU", "ZN", "GA", "RB", "SR", "Y", "ZR", "MO", "RU", "RH", "PD", "AG",
    "CD", "IN", "SN", "SB", "CS", "BA", "LA", "CE", "PR", "ND", "PM", "SM",
    "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB", "LU", "HF", "TA", "W",
    "RE", "OS", "IR", "PT", "AU", "HG", "TL", "PB", "BI", "TH", "PA", "U",
}
POSITION_PATTERN = re.compile(r"^(-?\d+)([A-Za-z]?)$")
ResidueKey = tuple[str, int, str]
FIRST_COORDINATION_THRESHOLD = 3.0
SECOND_COORDINATION_THRESHOLD = 5.0


class MutationOutsideMetalCoordinationError(ValueError):
    pass


@dataclass(frozen=True)
class MetalCoordinationSite:

    atom_serial: int
    element: str
    first_shell: frozenset[ResidueKey]
    second_shell: frozenset[ResidueKey]

    @property
    def coordinated_residues(self) -> frozenset[ResidueKey]:
        return self.first_shell | self.second_shell


def parse_position(value: str) -> tuple[int, str]:
    match = POSITION_PATTERN.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"Invalid PDB residue position: {value}")
    return int(match.group(1)), match.group(2).upper() or " "


def has_complete_backbone(residue) -> bool:
    return {"N", "CA", "C", "O"}.issubset(atom.id for atom in residue.get_atoms())


def residue_key(residue) -> ResidueKey:
    return residue.get_parent().id, residue.id[1], residue.id[2].strip()


def find_target_residue(model, chain_id: str, position: str, expected_resname: str):
    residue_number, insertion_code = parse_position(position)
    if chain_id not in model:
        raise ValueError(f"Chain {chain_id!r} is absent from the structure")
    residue_id = (" ", residue_number, insertion_code)
    chain = model[chain_id]
    if residue_id not in chain:
        raise ValueError(f"Residue {chain_id}:{position} is absent from the structure")
    residue = chain[residue_id]
    observed = residue.get_resname().upper()
    if observed != expected_resname.upper():
        raise ValueError(
            f"Structure residue mismatch at {chain_id}:{position}; "
            f"expected {expected_resname.upper()}, observed {observed}"
        )
    if not has_complete_backbone(residue):
        raise ValueError(f"Target residue {chain_id}:{position} has an incomplete backbone")
    return residue


def select_nearest_residues(model, target_residue, residue_count: int) -> list:
    if residue_count < 1:
        raise ValueError("residue_count must be at least 1")
    target_coord = target_residue["CA"].coord
    candidates: list[tuple[float, tuple[str, int, str], object]] = []
    for chain in model:
        for residue in chain:
            if residue is target_residue:
                continue
            if residue.get_resname().upper() not in STANDARD_AMINO_ACIDS:
                continue
            if not has_complete_backbone(residue):
                continue
            distance = float(np.linalg.norm(residue["CA"].coord - target_coord))
            candidates.append((distance, residue_key(residue), residue))
    candidates.sort(key=lambda item: (item[0], item[1]))
    if len(candidates) < residue_count - 1:
        raise ValueError(
            f"Structure contains only {len(candidates) + 1} usable amino acids; "
            f"{residue_count} are required"
        )
    selected = [target_residue, *(item[2] for item in candidates[: residue_count - 1])]
    return sorted(selected, key=residue_key)


def _atom_line_map(pdb_path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")):
                try:
                    mapping[int(line[6:11])] = line if line.endswith("\n") else line + "\n"
                except ValueError:
                    continue
    return mapping


def _non_hydrogen_coordinates(residue) -> np.ndarray:
    coordinates = [
        atom.coord
        for atom in residue.get_atoms()
        if (atom.element or "").strip().upper() not in {"H", "D"}
    ]
    return np.asarray(coordinates, dtype=float)


def find_metal_coordination_sites(
    model,
    first_coordination_threshold: float = FIRST_COORDINATION_THRESHOLD,
    second_coordination_threshold: float = SECOND_COORDINATION_THRESHOLD,
) -> list[MetalCoordinationSite]:
    if first_coordination_threshold <= 0 or second_coordination_threshold <= 0:
        raise ValueError("Coordination distance thresholds must be positive")

    amino_acid_coordinates = {
        residue_key(residue): coordinates
        for chain in model
        for residue in chain
        if is_aa(residue, standard=False)
        if (coordinates := _non_hydrogen_coordinates(residue)).size
    }
    sites: list[MetalCoordinationSite] = []
    for chain in model:
        for residue in chain:
            for atom in residue.get_atoms():
                element = (atom.element or "").strip().upper()
                if element not in METAL_ELEMENTS:
                    continue
                metal_coordinate = atom.coord
                first_shell = frozenset(
                    key
                    for key, coordinates in amino_acid_coordinates.items()
                    if float(np.linalg.norm(coordinates - metal_coordinate, axis=1).min())
                    < first_coordination_threshold
                )
                if first_shell:
                    first_shell_coordinates = np.concatenate(
                        [amino_acid_coordinates[key] for key in first_shell], axis=0
                    )
                    second_shell = frozenset(
                        key
                        for key, coordinates in amino_acid_coordinates.items()
                        if key not in first_shell
                        and float(
                            np.linalg.norm(
                                coordinates[:, np.newaxis, :]
                                - first_shell_coordinates[np.newaxis, :, :],
                                axis=2,
                            ).min()
                        )
                        < second_coordination_threshold
                    )
                else:
                    second_shell = frozenset()
                sites.append(
                    MetalCoordinationSite(
                        atom_serial=atom.get_serial_number(),
                        element=element,
                        first_shell=first_shell,
                        second_shell=second_shell,
                    )
                )
    return sites


def mutation_coordination_metals(
    pdb_path: Path,
    target_residue_key: ResidueKey,
    specified_metal: str,
    first_coordination_threshold: float = FIRST_COORDINATION_THRESHOLD,
    second_coordination_threshold: float = SECOND_COORDINATION_THRESHOLD,
) -> list[MetalCoordinationSite]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    try:
        model = next(structure.get_models())
    except StopIteration as exc:
        raise ValueError(f"PDB structure contains no models: {pdb_path}") from exc

    related_sites = [
        site
        for site in find_metal_coordination_sites(
            model,
            first_coordination_threshold,
            second_coordination_threshold,
        )
        if target_residue_key in site.coordinated_residues
    ]
    requested_element = specified_metal.strip().upper()
    if not any(site.element == requested_element for site in related_sites):
        return []
    return related_sites


def coordination_metal_lines(
    metal_source_path: Path,
    target_residue_key: ResidueKey,
    specified_metal: str,
    first_coordination_threshold: float,
    second_coordination_threshold: float,
) -> list[str]:
    sites = mutation_coordination_metals(
        metal_source_path,
        target_residue_key,
        specified_metal,
        first_coordination_threshold,
        second_coordination_threshold,
    )
    if not sites:
        raise MutationOutsideMetalCoordinationError(
            f"mutation residue is not in the first or second coordination shell "
            f"of the specified {specified_metal.upper()} ion"
        )
    line_map = _atom_line_map(metal_source_path)
    selected_lines: list[str] = []
    for site in sites:
        line = line_map.get(site.atom_serial)
        if line is None:
            raise ValueError(
                f"Cannot recover PDB line for metal atom serial {site.atom_serial} "
                f"in {metal_source_path}"
            )
        selected_lines.append(line)
    return selected_lines


def prepare_local_structure(
    full_pdb_path: Path,
    metal_source_path: Path,
    output_path: Path,
    chain_id: str,
    pdb_position: str,
    expected_amino_acid: str,
    specified_metal: str,
    residue_count: int = 128,
    first_coordination_threshold: float = FIRST_COORDINATION_THRESHOLD,
    second_coordination_threshold: float = SECOND_COORDINATION_THRESHOLD,
) -> tuple[int, int]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(full_pdb_path.stem, str(full_pdb_path))
    try:
        model = next(structure.get_models())
    except StopIteration as exc:
        raise ValueError(f"PDB structure contains no models: {full_pdb_path}") from exc

    expected_resname = ONE_TO_THREE[expected_amino_acid.upper()]
    target_residue = find_target_residue(
        model, chain_id, pdb_position, expected_resname
    )
    selected_residues = select_nearest_residues(model, target_residue, residue_count)
    line_map = _atom_line_map(full_pdb_path)
    atom_lines: list[str] = []
    for residue in selected_residues:
        for atom in residue.get_atoms():
            line = line_map.get(atom.get_serial_number())
            if line is None:
                raise ValueError(
                    f"Cannot recover PDB line for atom serial {atom.get_serial_number()} "
                    f"in {full_pdb_path}"
                )
            atom_lines.append(line)

    metal_lines = coordination_metal_lines(
        metal_source_path,
        residue_key(target_residue),
        specified_metal,
        first_coordination_threshold,
        second_coordination_threshold,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.writelines(atom_lines)
        handle.writelines(metal_lines)
        handle.write("END\n")
    return len(selected_residues), len(metal_lines)
