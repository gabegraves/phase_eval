"""U14 typed composition input: implied composition, and measurements through the real routes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from pymatgen.core import Composition, Element
from test_labels import RATER_A, RATER_B, Bench, assigned, bench  # noqa: F401  (pytest fixtures)
from test_tree import OWNER, World

from phaseeval.api.auth import CaseOperation, PrincipalAuthenticator
from phaseeval.workbench import composition_routes
from phaseeval.workbench.composition import CompositionMeasurementRequest, implied_composition

TOKENS = {"owner-1": ("user", ["owner"]), "agent-1": ("agent", [])}


@pytest.fixture
def client(tmp_path: Path) -> tuple[TestClient, World]:
    world = World(tmp_path / "store")
    principals = tmp_path / "principals.json"
    principals.write_text(
        json.dumps(
            {
                "schema_version": "axa.api_principals.v1",
                "principals": [
                    {
                        "principal_id": pid,
                        "principal_type": kind,
                        "credential_sha256": hashlib.sha256(f"token-{pid}".encode()).hexdigest(),
                        "roles": roles,
                    }
                    for pid, (kind, roles) in TOKENS.items()
                ],
            }
        )
    )
    app = FastAPI()
    app.include_router(composition_routes.router)
    app.state.workbench_store = world.store
    app.state.attempt_authenticator = PrincipalAuthenticator.from_file(principals)
    return TestClient(app), world


def bearer(principal_id: str) -> dict[str, str]:
    return {"Authorization": f"Bearer token-{principal_id}"}


XRF = {"basis": "weight", "method": "XRF", "fractions": {"Ni": 0.39}, "source": "XRF report R-17, p. 2"}


def test_a_one_to_one_molar_mixture_implies_its_atom_counts() -> None:
    nio, li2co3 = Composition("NiO").weight, Composition("Li2CO3").weight
    total = nio + li2co3
    implied = implied_composition([("P1", "NiO", 100 * nio / total), ("P2", "Li2CO3", 100 * li2co3 / total)])
    # One NiO and one Li2CO3: 8 atoms, Ni 1, Li 2, C 1, O 4.
    assert implied.status == "computed"
    assert implied.atomic_fractions == pytest.approx({"C": 0.125, "Li": 0.25, "Ni": 0.125, "O": 0.5})
    assert implied.weight_fractions["Ni"] == pytest.approx(float(Element("Ni").atomic_mass) / total)
    assert "not the specimen's composition" in implied.basis

    restricted = implied_composition([("P1", "NiO", 50.0), ("P2", "Li2CO3", 50.0)], restrict_to=("Ni", "C"))
    assert set(restricted.weight_fractions) == {"C", "Ni"}
    assert sum(restricted.atomic_fractions.values()) == pytest.approx(1.0)
    assert "reported elements C, Ni only" in restricted.basis


def test_a_phase_without_a_weight_or_formula_is_named_and_nothing_is_renormalized() -> None:
    implied = implied_composition([("P1", "NiO", 60.0), ("P2", "Li2CO3", None), ("P3", None, 5.0), ("P4", "Qq2", 1.0)])
    assert implied.status == "not_computable"
    assert (implied.phases_without_weight, implied.phases_without_formula) == (("P2",), ("P3", "P4"))
    assert implied.weight_fractions == {} and implied.atomic_fractions == {}
    assert "P2" in implied.reason and "P3, P4" in implied.reason


@pytest.mark.parametrize("weight", [-10.0, 110.0, float("nan"), float("inf")])
def test_an_out_of_range_weight_percent_is_refused(weight: float) -> None:
    with pytest.raises(ValueError, match="P1: weight percent"):
        implied_composition([("P1", "NiO", weight), ("P2", "TiO2", 50.0)])


@pytest.mark.parametrize(
    "fractions", [{}, {"Ni": -0.1}, {"Ni": float("nan")}, {"Ni": float("inf")}, {"Ni": 0.7, "O": 0.6}]
)
def test_a_measurement_with_bad_fractions_is_refused(fractions: dict) -> None:
    with pytest.raises(ValidationError):
        CompositionMeasurementRequest.model_validate(XRF | {"fractions": fractions})


def test_a_measurement_without_a_source_is_refused_and_nothing_is_stored(client) -> None:
    http, world = client
    url = f"/api/attempts/roots/{world.root}/composition"

    assert http.post(url, json=XRF | {"source": "  "}, headers=bearer("owner-1")).status_code == 422
    assert http.post(url, json={k: v for k, v in XRF.items() if k != "source"}, headers=bearer("owner-1")).status_code == 422
    assert http.get(url, headers=bearer("owner-1")).json() == []
    assert not (world.store.root / "compositions").exists()


def test_a_sourced_measurement_is_stored_with_the_server_stamped_principal(client) -> None:
    http, world = client
    url = f"/api/attempts/roots/{world.root}/composition"

    assert http.post(url, json=XRF | {"entered_by": "someone-else"}, headers=bearer("owner-1")).status_code == 422
    created = http.post(url, json=XRF, headers=bearer("owner-1"))
    assert created.status_code == 201
    assert http.get(url, headers=bearer("owner-1")).json() == [created.json()]
    assert created.json()["entered_by"] == "owner-1"
    assert created.json()["fractions"] == {"Ni": 0.39}


def test_composition_routes_refuse_no_bearer_and_an_agent(client) -> None:
    http, world = client
    url = f"/api/attempts/roots/{world.root}/composition"

    assert http.get(url).status_code == 401
    assert http.post(url, json=XRF).status_code == 401
    assert http.post("/api/attempts/implied-composition", json={"phases": []}).status_code == 401
    assert http.post(url, json=XRF, headers=bearer("agent-1")).status_code == 403
    assert http.get(url, headers=bearer("agent-1")).status_code == 403


def test_the_implied_route_computes_from_posted_answer_phases(client) -> None:
    http, _ = client
    body = {"phases": [{"phase_id": "P1", "formula": "NiO", "weight_percent": 100.0}]}
    response = http.post("/api/attempts/implied-composition", json=body, headers=bearer("owner-1"))
    assert response.status_code == 200
    assert response.json()["atomic_fractions"] == pytest.approx({"Ni": 0.5, "O": 0.5})
    body["phases"][0]["weight_percent"] = -10.0
    assert http.post("/api/attempts/implied-composition", json=body, headers=bearer("owner-1")).status_code == 422


def test_a_rater_reads_only_their_own_measurements_until_the_reveal(assigned: Bench) -> None:  # noqa: F811
    url = f"/roots/{assigned.root}/composition"
    for rater, nickel in ((RATER_A, 0.30), (RATER_B, 0.40)):
        assigned.store.grant(OWNER, assigned.root, rater.principal_id, (CaseOperation.ACTION,))
        assert assigned.post(rater, url, XRF | {"fractions": {"Ni": nickel}}).status_code == 201

    def entered(principal) -> list[str]:
        return sorted(m["entered_by"] for m in assigned.get(principal, url).json())

    assert entered(RATER_A) == ["rater-a"]
    assert entered(RATER_B) == ["rater-b"]
    assert entered(OWNER) == ["rater-a", "rater-b"]  # the admin owner is unchanged
    assigned.sign_all(RATER_A)
    assert entered(RATER_A) == ["rater-a"]  # rater-b has not signed off, so no reveal yet
    assigned.sign_all(RATER_B)
    assert entered(RATER_A) == ["rater-a", "rater-b"]
    assert entered(RATER_B) == ["rater-a", "rater-b"]
