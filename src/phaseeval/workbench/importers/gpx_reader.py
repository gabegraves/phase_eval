"""Read one finished GSAS-II project and print it as JSON. Runs only inside the sandbox.

A .gpx is a sequence of Python pickles, so opening one can run arbitrary code. This script is
started only by importers.gsasii.read_gpx under bubblewrap, never imported by the server. It is
standalone on purpose: stdlib, numpy and GSAS-II only, located through GSASII_SOURCE. GSAS-II's
own console output goes to stderr; stdout carries exactly one JSON object.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys


def _num(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _series(values: object) -> list[float | None]:
    filled = values.filled(float("nan")) if hasattr(values, "filled") else values  # numpy masked array
    return [_num(v) for v in filled]


# GSAS-II instrument type -> what the histogram's native X is (GSASIIscriptable.getdata: X may be
# 2theta or TOF). Angle-dispersive types end in C (constant wavelength) or B (pink beam).
_X_QUANTITY = {"C": "two_theta_deg", "B": "two_theta_deg", "T": "time_of_flight_us", "E": "energy_kev"}
FRACTION_BASIS = (
    "GSAS-II ComputeMassFracs over the saved phase fractions, normalized over the phases linked to "
    "the histogram. Its s.u. is GSAS-II's propagation through the saved covariance and is kept "
    "whenever any linked phase fraction varied or depended on a varied parameter: a fixed scale "
    "still has an uncertain normalized fraction when another scale moves its denominator. "
    "scale_status names this phase's own scale parameter (varied, dependent or fixed)."
)


def fraction_record(
    value: object, su: object, own_scale: str, scales: set[str], vary: set[str], dependent: set[str], has_covariance: bool
) -> dict[str, object]:
    """One derived mass fraction. Pure, so its rule is testable without opening a project."""
    moving = vary | dependent
    scale_status = "varied" if own_scale in vary else "dependent" if own_scale in dependent else "fixed"
    native_su = _num(su)
    propagated = has_covariance and (bool(scales & moving) or bool(native_su))
    return {
        "status": "derived",
        "scale_status": scale_status,
        "value": _num(value),
        # all scales fixed: GSAS-II returns 0, which is not established exactness
        "su": native_su if propagated else None,
        "detail": "",
    }


def _read(path: str) -> dict[str, object]:
    from GSASII import GSASIIscriptable as G2sc

    project = G2sc.G2Project(gpxfile=path)
    try:
        covariance = project["Covariance"]["data"]
    except (KeyError, TypeError):
        covariance = {}
    covariance = covariance if isinstance(covariance, dict) else {}
    vary = set(covariance.get("varyList") or [])
    dep_parms = covariance.get("depParmDict")
    dependent = set(dep_parms) if isinstance(dep_parms, dict) else set()  # constrained variables
    has_covariance = covariance.get("covMatrix") is not None
    rvals = covariance.get("Rvals", {}) or {}
    try:
        saved_using = project["Controls"]["data"].get("LastSavedUsing")
    except (KeyError, TypeError, AttributeError):
        saved_using = None
    phases = []
    for phase in project.phases():
        refined_cell = has_covariance and any(f"{phase.id}::A{i}" in vary | dependent for i in range(6))
        cell, esd = phase.get_cell_and_esd() if refined_cell else (phase.get_cell(), {})
        phases.append(
            {
                "name": phase.name,
                "space_group": phase.data["General"].get("SGData", {}).get("SpGrp"),
                "cell": {k: _num(v) for k, v in cell.items()},
                "cell_refined": refined_cell,
                # s.u. only from the covariance of a refined cell; GSAS-II returns 0 otherwise
                "cell_esd": {k: _num(v) for k, v in esd.items()} if refined_cell else {},
                # General doPawley: the project's own record that the phase is a Pawley extraction
                "pawley": bool(phase.data["General"].get("doPawley", False)),
            }
        )
    histograms = []
    for hist in project.histograms():
        instrument = hist.InstrumentParameters
        itype = str((instrument.get("Type") or [None, None])[1] or "")
        hid = hist.data["data"][0].get("hId")
        lam = instrument.get("Lam") or instrument.get("Lam1")
        try:
            computed, error = hist.ComputeMassFracs(), None
        except Exception as exc:  # recorded per phase below, never collapsed to "not refined"
            computed, error = {}, f"{type(exc).__name__}: {exc}"
        fractions = {}
        linked = [p for p in project.phases() if hist.name in p.data.get("Histograms", {})]
        scales = {f"{p.id}:{hid}:Scale" for p in linked}
        # The histogram-phase LeBail flag (GSASIIstrMain.CheckLeBail reads the same key)
        le_bail = [p.name for p in linked if (p.data["Histograms"][hist.name] or {}).get("LeBail")]
        for phase in linked:
            if error:
                fractions[phase.name] = {"status": "error", "scale_status": None, "value": None, "su": None, "detail": error}
            elif phase.name not in computed:
                fractions[phase.name] = {"status": "not_computable", "scale_status": None, "value": None, "su": None, "detail": "no fraction returned"}
            else:
                value, su = computed[phase.name]
                fractions[phase.name] = fraction_record(
                    value, su, f"{phase.id}:{hid}:Scale", scales, vary, dependent, has_covariance
                )
        histograms.append(
            {
                "name": hist.name,
                "instrument_type": itype or None,
                "x_quantity": _X_QUANTITY.get(itype[2:3], "unknown"),
                "wavelength_angstrom": _num(lam[1]) if isinstance(lam, (list, tuple)) and len(lam) > 1 else None,
                "wr_percent": _num(hist.data["data"][0].get("wR")),
                "x": _series(hist.getdata("x")),
                "observed": _series(hist.getdata("yobs")),
                "calculated": _series(hist.getdata("ycalc")),
                "background": _series(hist.getdata("background")),
                "residual": _series(hist.getdata("residual")),
                "fraction_basis": FRACTION_BASIS,
                "mass_fractions": fractions,
                "le_bail_phases": le_bail,
            }
        )
    try:
        from GSASII import GSASIIpath

        reader_version = str(GSASIIpath.GetVersionNumber())
    except Exception:
        reader_version = None
    return {
        "reader_gsasii_version": reader_version,
        "saved_using": str(saved_using) if saved_using else None,
        "rwp_percent": _num(rvals.get("Rwp")),
        "gof": _num(rvals.get("GOF")),
        "converged": rvals.get("converged") if isinstance(rvals.get("converged"), bool) else None,
        "phases": phases,
        "histograms": histograms,
    }


def main(argv: list[str]) -> int:
    source = os.environ["GSASII_SOURCE"]
    sys.path[:0] = [source, os.path.join(source, "backcompat")]
    with contextlib.redirect_stdout(sys.stderr):
        result = _read(argv[1])
    sys.stdout.write(json.dumps(result, allow_nan=False))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
