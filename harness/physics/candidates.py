"""One line per candidate structure: id, formula, space group, crystal system and cell, read from each CIF."""

from __future__ import annotations

import argparse
from pathlib import Path

from casedata import INPUT_DIR, load_candidates


def cell_text(cell: dict) -> str:
    """Cell lengths, with b and c only when they differ from a; angles only when not 90."""
    a = cell["a"]
    parts = [f"a {a:.3f}" if a else "a ?"]
    for key in "bc":
        if not (a and cell[key] and abs(cell[key] - a) < 1e-4):
            parts.append(f"{key} {cell[key]:.3f}" if cell[key] else f"{key} ?")
    angles = [(key, cell[key]) for key in ("alpha", "beta", "gamma") if cell[key] and abs(cell[key] - 90) > 1e-6]
    if len(angles) == 3 and len({round(value, 4) for _, value in angles}) == 1:
        parts.append(f"angles {angles[0][1]:g}")
    else:
        parts += [f"{key} {value:g}" for key, value in angles]
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    rows = sorted(load_candidates(args.case, args.input_dir), key=lambda row: (row["formula"], row["candidate_id"]))
    print(f"{args.case}: {len(rows)} candidates, grouped by formula: id formula space-group (No.) crystal-system: cell "
          "in Angstrom (b and c shown when they differ from a) and degrees (angles shown when not 90)")
    for row in rows:
        number = f" ({row['sg_number']})" if row["sg_number"] else ""
        print(f"{row['candidate_id']} {row['formula']} {row['sg_symbol']}{number} {row['system']}: {cell_text(row['cell'])}")


if __name__ == "__main__":
    main()
