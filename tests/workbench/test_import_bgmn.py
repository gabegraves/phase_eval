"""BGMN/Profex import: saved BGMN outputs are read as the source report printed them.

The bundle is the Rietveld stage of the DEV006 example (harness/examples/dev006, public DARA scan
and COD structures), run by BGMN 4.2.23 from a relative control path and committed under
tests/workbench/fixtures/dev006_bgmn. Scenarios: statistics deleted from the .lst, a bundle of only
.sav/.lst/.dia, the recorded Q GOALs against normalized GEWICHT, a Profex archive with Windows line
endings, and bundles that are refused.
"""

from __future__ import annotations

import hashlib
import math
import re
import shutil
import zipfile
from pathlib import Path

import pytest

from phaseeval.workbench.evidence import Curve
from phaseeval.workbench.importers.bgmn import BundleError, Statistic, read_bundle

BUNDLE = Path(__file__).resolve().parent / "fixtures" / "dev006_bgmn"
STEM = "DEV006_rietveld"
CONTROL = f"{STEM}.sav"
PHASES = ("cod1010093", "cod9008283", "cod1526458")
# The .lst as BGMN printed it: global R values and the Q GOAL lines (Qcod1010093=0.3572+-0.0072).
RECORDED_RWP, RECORDED_REXP, RECORDED_RP, RECORDED_RPB = 3.52, 1.94, 2.75, 14.46
GOAL_PERCENT = (35.72, 43.3, 21.0)
GOAL_ERROR_PERCENT = (0.72, 1.1, 0.42)
# MeanValue(GEWICHT) 0.137255 / 0.166312 / 0.0806729, normalized: the derived estimate.
GEWICHT = (0.137255, 0.166312, 0.0806729)
DERIVED_PERCENT = tuple(100 * g / sum(GEWICHT) for g in GEWICHT)
NO_GOALS = re.compile(r"^(?:sum|Q\w+|GOAL\[\d+\])=.*\n", re.MULTILINE)


def _structures() -> dict[str, str]:
    """BGMN phase name -> the .str member that declares it."""
    found = {}
    for path in BUNDLE.glob("*.str"):
        name = re.match(r"PHASE=(\S+)", path.read_text())
        found[name[1]] = path.name
    return found


def _copy(tmp_path: Path, names: tuple[str, ...] | None = None) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for path in BUNDLE.iterdir():
        if names is None or path.name in names:
            shutil.copy(path, bundle / path.name)
    return bundle


def test_saved_bgmn_outputs_import_as_the_source_report_printed_them() -> None:
    analysis, blobs = read_bundle(BUNDLE)

    fit = analysis.fit
    assert fit.rwp == Statistic(value=RECORDED_RWP) and fit.rexp == Statistic(value=RECORDED_REXP)
    assert fit.rp == Statistic(value=RECORDED_RP) and fit.rpb == Statistic(value=RECORDED_RPB)
    assert fit.gof == pytest.approx(RECORDED_RWP / RECORDED_REXP)
    assert (fit.convergence, fit.profile) == ("unknown", "unknown")
    # the stage is read from the run: no STRUC file sets LeBail, so the intensities are computed
    assert fit.stage == "rietveld" and fit.stage_basis.startswith("no STRUC file sets LeBail")
    assert analysis.bgmn_version == "4.2.23" and fit.iterations == 106
    assert (fit.measured_points, fit.peaks, fit.parameters) == (8282, 461, 74)
    assert tuple(phase.name for phase in fit.phases) == PHASES
    nio, li2co3, bi2o3 = fit.phases
    assert (nio.spacegroup_no, li2co3.spacegroup_no, bi2o3.spacegroup_no) == (225, 15, 14)
    assert nio.cell.a == pytest.approx(4.178582) and nio.cell.a == nio.cell.b == nio.cell.c
    assert li2co3.cell.beta == pytest.approx(115.411) and li2co3.cell.alpha == li2co3.cell.gamma == 90.0
    assert analysis.control == CONTROL and analysis.missing_members == ()
    assert all(member.ref.sha256 in blobs for member in analysis.members.values())
    structures = _structures()
    for phase in fit.phases:
        assert phase.structure == structures[phase.name]


def test_the_recorded_goal_centers_and_errors_are_kept_as_printed() -> None:
    """The .sav defines Q<phase> = phase/sum over every phase, so the .lst's Q lines are the
    recorded weights, not normalized GEWICHT."""
    fit = read_bundle(BUNDLE)[0].fit

    assert fit.weight_basis == "recorded_goal"
    assert tuple(phase.weight_percent for phase in fit.phases) == pytest.approx(GOAL_PERCENT)
    assert tuple(phase.weight_percent_error for phase in fit.phases) == pytest.approx(GOAL_ERROR_PERCENT)


@pytest.mark.parametrize("defect", ["no_goals", "sum_omits_a_phase", "goal_printed_error"])
def test_without_validated_goals_normalized_gewicht_is_a_derived_estimate_with_no_error(
    tmp_path: Path, defect: str
) -> None:
    bundle = _copy(tmp_path)
    sav, lst = bundle / CONTROL, bundle / f"{STEM}.lst"
    if defect == "no_goals":
        sav.write_text(NO_GOALS.sub("", sav.read_text()))
    elif defect == "sum_omits_a_phase":
        sav.write_text(sav.read_text().replace("sum=cod1010093+cod9008283+cod1526458", "sum=cod1010093+cod9008283"))
    else:
        lst.write_text(lst.read_text().replace("Qcod1526458=0.2100+-0.0042", "Qcod1526458=ERROR"))

    fit = read_bundle(bundle)[0].fit

    assert fit.weight_basis == "derived_gewicht"
    assert tuple(phase.weight_percent for phase in fit.phases) == pytest.approx(DERIVED_PERCENT)
    assert all(phase.weight_percent_error is None for phase in fit.phases)
    assert "derived estimate with no recorded error" in fit.weight_detail


def test_a_lst_without_its_statistics_records_their_absence(tmp_path: Path) -> None:
    bundle = _copy(tmp_path)
    lst = bundle / f"{STEM}.lst"
    lst.write_text(re.sub(r"^Rp=.*$", "", lst.read_text(), flags=re.MULTILINE))

    fit = read_bundle(bundle)[0].fit

    assert fit.rwp.value is None and "prints no Rwp" in fit.rwp.absent
    assert fit.rexp.value is None and fit.gof is None
    assert tuple(phase.weight_percent for phase in fit.phases) == pytest.approx(GOAL_PERCENT)


def test_a_bundle_of_only_the_control_and_its_outputs_imports_and_names_what_is_missing(tmp_path: Path) -> None:
    bundle = _copy(tmp_path, (CONTROL, f"{STEM}.lst", f"{STEM}.dia"))

    analysis, _ = read_bundle(bundle)

    assert analysis.fit.rwp.value == RECORDED_RWP
    assert all(phase.structure is None for phase in analysis.fit.phases)
    missing = set(analysis.missing_members)
    assert "VAL[1]=DEV006.xy" in missing and f"OUTPUT={STEM}.par" in missing
    assert any(reference.startswith("STRUC[1]=") for reference in missing)


def test_members_take_the_role_of_the_control_key_that_names_them() -> None:
    analysis, _ = read_bundle(BUNDLE)
    roles = {name: (member.ref.role, member.named_by) for name, member in analysis.members.items()}

    assert roles["DEV006.xy"] == ("observed_pattern", ("VAL[1]",))
    assert roles[f"{STEM}.lst"] == ("native_lst", ("LIST",))
    assert roles[f"{STEM}.dia"] == ("native_dia", ("DIAGRAMM",))
    assert roles[f"{STEM}.par"] == ("native_par", ("OUTPUT",))
    assert roles["Aeris-fds-Pixcel1d-Medipix3.geq"] == ("raw_export", ("VERZERR",))
    assert roles["Aeris-fds-Pixcel1d-Medipix3.ger"] == ("raw_export", ())
    assert roles[CONTROL] == ("raw_export", ())
    structures = _structures()
    for n, phase in enumerate(analysis.fit.phases, start=1):
        assert roles[structures[phase.name]] == ("raw_export", (f"STRUC[{n}]",))


def test_the_imported_curve_recomputes_the_recorded_rp() -> None:
    analysis, blobs = read_bundle(BUNDLE)
    curve = Curve.parse_csv(blobs[analysis.fit.curve.sha256])

    residual = sum(abs(o - c) for o, c in zip(curve.observed, curve.calculated, strict=True))
    rp = 100 * residual / sum(curve.observed)

    assert len(curve.two_theta) == analysis.fit.measured_points
    assert curve.background is not None
    assert rp == pytest.approx(RECORDED_RP, abs=0.005)


def test_a_profex_archive_with_windows_line_endings_imports_the_same_fit(tmp_path: Path) -> None:
    archive = tmp_path / "project.zip"
    crlf: dict[str, bytes] = {}
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in sorted(BUNDLE.iterdir()):
            data = path.read_bytes()
            if path.suffix in (".sav", ".lst", ".str"):
                data = crlf[path.name] = data.replace(b"\n", b"\r\n")
            bundle.writestr(f"project/{path.name}", data)

    from_archive, _ = read_bundle(archive)
    from_directory, _ = read_bundle(BUNDLE)

    assert from_archive.fit == from_directory.fit
    assert from_archive.missing_members == () and set(from_archive.members) == set(from_directory.members)
    for name, data in crlf.items():  # CRLF is normalized for parsing only
        assert from_archive.members[name].ref.sha256 == hashlib.sha256(data).hexdigest()


def test_a_bundle_with_several_control_files_needs_one_named(tmp_path: Path) -> None:
    """The harness writes one .sav per stage into one fit directory."""
    bundle = _copy(tmp_path)
    shutil.copy(bundle / CONTROL, bundle / "DEV006_pawley.sav")

    with pytest.raises(BundleError, match="--control"):
        read_bundle(bundle)
    with pytest.raises(BundleError, match="no control file"):
        read_bundle(bundle, control="other.sav")
    assert read_bundle(bundle, control=CONTROL)[0].fit == read_bundle(BUNDLE)[0].fit


@pytest.mark.parametrize(
    "defect", ["two_controls", "missing_list", "missing_diagram", "climbing_member", "repeated_member"]
)
def test_a_bundle_without_its_named_outputs_or_unsafe_is_refused(tmp_path: Path, defect: str) -> None:
    bundle = tmp_path / "bundle"
    shutil.copytree(BUNDLE, bundle)
    if defect == "two_controls":
        shutil.copy(bundle / CONTROL, bundle / "second.sav")
    elif defect == "missing_list":
        (bundle / f"{STEM}.lst").unlink()
    elif defect == "missing_diagram":
        (bundle / f"{STEM}.dia").unlink()
    else:
        bundle = tmp_path / "bundle.zip"
        with zipfile.ZipFile(bundle, "w") as archive:
            if defect == "climbing_member":
                archive.writestr("../outside.sav", (BUNDLE / CONTROL).read_bytes())
            else:
                for path in sorted(BUNDLE.iterdir()):
                    archive.writestr(f"project/{path.name}", path.read_bytes())
                archive.writestr(f"other/{STEM}.lst", b"Rwp=1.00%\n")

    with pytest.raises(BundleError):
        read_bundle(bundle)


# The .sav and .lst of a real DEV006 fit at texture order 10. BGMN printed Qcod1010093=0+-15,
# Qcod9008283=0+-10 and Qcod1526458=0.2+-4.4; MeanValue(GEWICHT) is 0.134234 / 0.201497 / 0.0845181.
TEX10 = Path(__file__).resolve().parent / "fixtures" / "dev006_bgmn_tex10"
TEX10_GEWICHT = (0.134234, 0.201497, 0.0845181)
TEX10_PERCENT = (31.94, 47.95, 20.11)


def _tex10_bundle(tmp_path: Path) -> Path:
    bundle = _copy(tmp_path, (f"{STEM}.dia",))
    for path in TEX10.iterdir():
        shutil.copy(path, bundle / path.name)
    return bundle


def _with_grain_diameter(bundle: Path, line: str = "d=5.0", my: tuple[str, ...] = ()) -> None:
    sav, lst = bundle / CONTROL, bundle / f"{STEM}.lst"
    sav.write_text(sav.read_text().replace("NTHREADS=", f"{line}\nNTHREADS=", 1))
    if my:
        values = iter(my)
        lst.write_text(re.sub(r"^GEWICHT=", lambda _: f"my={next(values)}\nGEWICHT=", lst.read_text(), count=len(my), flags=re.M))


def test_a_goal_rounded_to_its_esd_is_not_recorded_as_an_amount(tmp_path: Path) -> None:
    fit = read_bundle(_tex10_bundle(tmp_path))[0].fit

    assert fit.weight_basis == "derived_gewicht"
    assert tuple(phase.weight_percent for phase in fit.phases) == pytest.approx(TEX10_PERCENT, abs=0.01)
    assert all(phase.weight_percent_error is None for phase in fit.phases)
    assert "Qcod1010093=0+-15" in fit.weight_detail and "rounds a GOAL to its esd" in fit.weight_detail
    assert "derived estimate with no recorded error" in fit.weight_detail


@pytest.mark.parametrize(
    ("printed", "rounded"), [("0.2100+-0.2100", True), ("-0.01+-0.02", True), ("0+-0", True), ("0.2100+-0.2000", False)]
)
def test_a_goal_is_rounded_when_its_esd_reaches_its_center(tmp_path: Path, printed: str, rounded: bool) -> None:
    bundle = _copy(tmp_path)
    lst = bundle / f"{STEM}.lst"
    lst.write_text(lst.read_text().replace("Qcod1526458=0.2100+-0.0042", f"Qcod1526458={printed}"))

    fit = read_bundle(bundle)[0].fit

    assert fit.weight_basis == ("derived_gewicht" if rounded else "recorded_goal")
    assert tuple(phase.weight_percent for phase in fit.phases)[:2] == pytest.approx(
        DERIVED_PERCENT[:2] if rounded else GOAL_PERCENT[:2]
    )


def test_a_rounded_goal_with_a_grain_diameter_applies_the_correction_to_gewicht(tmp_path: Path) -> None:
    bundle = _tex10_bundle(tmp_path)
    my = (0.02, 0.01, 0.05)
    _with_grain_diameter(bundle, my=tuple(f"{value}+-0.000001" for value in my))

    fit = read_bundle(bundle)[0].fit

    corrected = [g * math.exp(m * 5.0 * 3 / 4) for g, m in zip(TEX10_GEWICHT, my, strict=True)]
    assert fit.weight_basis == "derived_gewicht" and "exp(my" in fit.weight_detail
    assert tuple(phase.weight_percent for phase in fit.phases) == pytest.approx(
        tuple(100 * value / sum(corrected) for value in corrected)
    )
    assert all(phase.weight_percent_error is None for phase in fit.phases)
    assert "derived estimate with no recorded error" in fit.weight_detail


@pytest.mark.parametrize(
    ("line", "my", "reason"),
    [
        ("d=5.0", (), "no usable pair for cod1010093, cod9008283, cod1526458"),
        ("d=5.0", ("0.02+-0.000001", "0.01+-0.000001"), "no usable pair for cod1526458"),
        ("PARAM=d=5.0_1^10", ("0.02", "0.01", "0.05"), "not one positive number"),
        ("d=0", ("0.02", "0.01", "0.05"), "not one positive number"),
    ],
)
def test_a_grain_diameter_without_a_usable_correction_leaves_the_amounts_unknown(
    tmp_path: Path, line: str, my: tuple[str, ...], reason: str
) -> None:
    bundle = _tex10_bundle(tmp_path)
    _with_grain_diameter(bundle, line, my)

    fit = read_bundle(bundle)[0].fit

    assert fit.weight_basis == "unavailable" and reason in fit.weight_detail
    assert all(phase.weight_percent is None and phase.weight_percent_error is None for phase in fit.phases)


@pytest.mark.parametrize("line", ["% d=5.0", "// d=5.0", "sum_d=5.0"])
def test_a_commented_or_longer_name_is_not_a_grain_diameter(tmp_path: Path, line: str) -> None:
    bundle = _tex10_bundle(tmp_path)
    _with_grain_diameter(bundle, line)

    fit = read_bundle(bundle)[0].fit

    assert fit.weight_basis == "derived_gewicht"
    assert tuple(phase.weight_percent for phase in fit.phases) == pytest.approx(TEX10_PERCENT, abs=0.01)


def test_well_determined_goals_with_a_grain_diameter_stay_recorded(tmp_path: Path) -> None:
    """The GOAL itself applies exp(my*d*3/4), so its printed center needs no correction."""
    bundle = _copy(tmp_path)
    _with_grain_diameter(bundle)

    fit = read_bundle(bundle)[0].fit

    assert fit.weight_basis == "recorded_goal"
    assert tuple(phase.weight_percent for phase in fit.phases) == pytest.approx(GOAL_PERCENT)
    assert tuple(phase.weight_percent_error for phase in fit.phases) == pytest.approx(GOAL_ERROR_PERCENT)
