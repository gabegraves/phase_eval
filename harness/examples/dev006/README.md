# DEV006 example

A case folder for the four tools in `harness/physics`, built from public data only. `run.sh` runs them in order. Set `OUT` to choose the output folder. A run label cannot be reused, so clear `OUT` before running again.

## Pattern

`input/DEV006.xy` is the DARA benchmark scan of an unheated NiO, Li2CO3 and Bi2O3 mix (Zenodo 10.5281/zenodo.17410051, CC BY 4.0). It is the same file as `data/workbench/cases/DEV006/DEV006.xy`. The manifest lists the three precursors without masses, because the public record does not give them.

## Candidate structures

The three CIFs come from the Crystallography Open Database, whose data are dedicated to the public domain (CC0).

| candidate_id | COD id | Phase | Space group | Source |
|---|---|---|---|---|
| cod_1010093 | 1010093 | NiO | Fm-3m | J. Am. Chem. Soc. 55 (1933) |
| cod_9008283 | 9008283 | Li2CO3 | C2/c | Z. Kristallogr. 150 (1979) |
| cod_1526458 | 1526458 | Bi2O3 (alpha) | P2_1/c | Powder Diffraction 16 (2001) |

Download URL for each: `https://www.crystallography.net/cod/<COD id>.cif`.

## Result on f65e3e5

`refine.py` finished all four stages. Final Rietveld Rwp 3.52% (Pawley 3.31%). This is a fit to the precursor structures, not a graded answer.
