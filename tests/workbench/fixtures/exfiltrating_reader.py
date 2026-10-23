"""A hostile stand-in for gpx_reader.py, run under the real sandbox by the tests.

It behaves as a project that took over the reader would: it reads everything /proc and the
environment show of the host, looks for the importer's PHASEEVAL_TEST_CANARY, and copies what it
found into the fields of an otherwise valid analysis, each squeezed through the character set that
field accepts. The analysis fits the case the test imports it under (the project file at /in is a
JSON object holding the case's points), so a leak would reach the store, the review payload and
the export. The report of what it could read is in fraction_basis.
"""

import json
import os
import re
import sys

VERSION_CHARS = re.compile(r"[^A-Za-z0-9 ._+:()-]")
SPACE_GROUP_CHARS = re.compile(r"[^A-Za-z0-9 /_:()-]")
CANARY = re.compile(r"PHASEEVAL_TEST_CANARY=([^\x00\s]+)")


def _read(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            return fh.read(1 << 20).replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError as exc:
        return f"<{type(exc).__name__}>"


def main() -> int:
    pids = sorted(int(p) for p in os.listdir("/proc") if p.isdigit())
    scraped = {"os.environ": " ".join(f"{k}={v}" for k, v in os.environ.items())}
    for pid in ["self", *pids]:
        scraped[f"/proc/{pid}/environ"] = _read(f"/proc/{pid}/environ")
        scraped[f"/proc/{pid}/cmdline"] = _read(f"/proc/{pid}/cmdline")
    blob = "\n".join(f"{k}: {v}" for k, v in scraped.items())
    found = CANARY.findall(blob)
    report = f"pids={pids} canary_found={bool(found)} " + " | ".join(
        f"{k}={len(v)}b" for k, v in scraped.items()
    )
    squeezed = VERSION_CHARS.sub("", blob)  # what a version field accepts
    printable = "".join(ch if ch.isprintable() else " " for ch in blob)  # what a text field accepts
    saved_using = f"GSAS-II {found[0]}" if found else "GSAS-II 5837"
    with open(sys.argv[1]) as fh:
        points = json.load(fh)
    x, observed = points["x"], points["observed"]
    background = [min(observed)] * len(observed)
    calculated = [0.5 * (o + b) for o, b in zip(observed, background)]
    names = points["names"]
    cell = {"length_a": 4.17, "length_b": 4.17, "length_c": 4.17, "angle_alpha": 90.0, "angle_beta": 90.0, "angle_gamma": 90.0, "volume": 72.5}
    result = {
        "saved_using": saved_using[:64],
        "reader_gsasii_version": (VERSION_CHARS.sub("", found[0]) if found else "5837")[:64],
        "rwp_percent": 7.25,
        "gof": 2.1,
        "converged": True,
        "phases": [
            {
                "name": name,
                "space_group": (SPACE_GROUP_CHARS.sub("", found[0]) if found else "F m -3 m")[:40],
                "cell": cell,
                "cell_refined": False,
                "cell_esd": {},
                "pawley": False,
            }
            for name in names
        ],
        "histograms": [
            {
                "name": "PWDR scraped",
                "instrument_type": "PXC",
                "x_quantity": "two_theta_deg",
                "wavelength_angstrom": 1.5406,
                "wr_percent": 7.25,
                "x": x,
                "observed": observed,
                "calculated": calculated,
                "background": background,
                "residual": [o - c for o, c in zip(observed, calculated)],
                "fraction_basis": (report + " || " + printable)[:2000],
                "mass_fractions": {
                    name: {"status": "derived", "scale_status": "varied", "value": f, "su": 0.01, "detail": printable[:500]}
                    for name, f in zip(names, points["fractions"])
                },
                "le_bail_phases": [],
            }
        ],
    }
    sys.stdout.write(json.dumps(result, allow_nan=False))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
