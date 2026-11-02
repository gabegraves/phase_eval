# DEV006 example

A case folder for the four tools in `harness/physics`, built from public data only. `run.sh` runs them in order. Set `OUT` to choose the output folder. A run label cannot be reused, so clear `OUT` before running again.

## Pattern

`input/DEV006.xy` is the DARA benchmark scan of an unheated NiO, Li2CO3 and Bi2O3 mix (Zenodo 10.5281/zenodo.17410051, CC BY 4.0). It is the same file as `data/workbench/cases/DEV006/DEV006.xy`. The manifest lists the three precursors without masses, because the public record does not give them.

## Candidate structures

The four CIFs come from the Crystallography Open Database, whose data are dedicated to the public domain (CC0).

| candidate_id | COD id | Phase | Space group | Source |
|---|---|---|---|---|
| cod_1010093 | 1010093 | NiO | Fm-3m | J. Am. Chem. Soc. 55 (1933) |
| cod_9008283 | 9008283 | Li2CO3 | C2/c | Z. Kristallogr. 150 (1979) |
| cod_1526458 | 1526458 | Bi2O3 (alpha) | P2_1/c | Powder Diffraction 16 (2001) |
| cod_9004677 | 9004677 | Bi2O2CO3 (bismutite) | Imm2 | Can. Mineral. 40 (2002) |

Bismutite is not a precursor. It is in the library because the pattern contains it: see the bismutite result below.

Download URL for each: `https://www.crystallography.net/cod/<COD id>.cif`.

## Result on f65e3e5

`refine.py` finished all four stages. Final Rietveld Rwp 3.52% (Pawley 3.31%). This is a fit to the precursor structures, not a graded answer.

## Bismutite result (2026-10-28)

The three-phase fit leaves residual peaks at 23.9, 30.3 and 32.7 degrees that no precursor explains. They match bismutite (011), (013) and (110). Bismutite commonly forms on Bi2O3 stored in air. Adding `cod_9004677` to the three precursors gives:

| | Precursors only | Precursors + bismutite |
|---|---|---|
| Rietveld Rwp (Rexp 1.94%) | 3.52% | 2.49% |
| GoF | 1.81 | 1.28 |
| Li2CO3 cell shift a / c / beta | +0.77% / +0.57% / +0.51% | +0.05% / -0.02% / -0.14% |
| Li2CO3 | 43.5 wt% | 39.0 wt% |
| Bi2O2CO3 | n/a | 2.70 +- 0.05 wt% |

With bismutite, the residuals at its line positions drop to near zero. The Li2CO3 cell returns to the CIF values, because Li2CO3 no longer stretches to absorb bismutite intensity. COD 2300744 (I4/mmm), an independent bismutite structure, gives the same result (Rwp 2.50%, 2.88 wt%). Other fourth-phase candidates have not been tested.

`run.sh` still refines the three precursors. To refine with bismutite, pass `--candidates cod_1010093 cod_9008283 cod_1526458 cod_9004677`.
