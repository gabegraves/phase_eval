"""Static principal authentication and case-scoped authorization."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from phaseeval.kernel.errors import Denied

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PRINCIPAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class AuthenticationError(PermissionError):
    """Raised when a request has no valid server-configured principal."""


class AuthorizationError(PermissionError):
    """Raised when a principal cannot perform an operation on a case."""


# Label-mode credentials (R33, Build D11): a rater or the owner in label mode. They read only the
# blind workbench routes and are refused the raw attempt views, which carry run identity.
LABEL_MODE_ROLES: frozenset[str] = frozenset({"rater", "owner_rater"})


class PrincipalType(StrEnum):
    USER = "user"
    AGENT = "agent"
    SERVICE = "service"
    # A model answering the frozen rubric over blind snapshots. It writes only model_judge
    # labels and is never an actor, an expert or an owner (workbench labels, PI directive).
    MODEL_JUDGE = "model_judge"


class CaseOperation(StrEnum):
    CREATE = "create"
    INSPECT = "inspect"
    ANALYZE = "analyze"
    ACTION = "action"
    FULFILL = "fulfill"
    RECONCILE = "reconcile"
    RESUME = "resume"
    EXPORT = "export"
    ARTIFACT = "artifact"
    WEBSOCKET = "websocket"


CASE_SCOPED_OPERATIONS = tuple(
    operation for operation in CaseOperation if operation is not CaseOperation.CREATE
)


@dataclass(frozen=True)
class Principal:
    principal_id: str
    principal_type: PrincipalType
    roles: frozenset[str]

    def __post_init__(self) -> None:
        if not _PRINCIPAL_ID.fullmatch(self.principal_id):
            raise ValueError("principal_id must be a safe identifier")
        if not isinstance(self.principal_type, PrincipalType):
            raise ValueError("principal_type must be recognized")

    def identity(self) -> dict[str, Any]:
        return {'principal_id': self.principal_id, 'principal_type': self.principal_type.value,
                'roles': sorted(self.roles)}


@dataclass(frozen=True, repr=False)
class _PrincipalCredential:
    principal: Principal
    credential_sha256: str


class PrincipalAuthenticator:
    """Authenticate bearer secrets against an external hash-only registry."""

    def __init__(self, credentials: tuple[_PrincipalCredential, ...]) -> None:
        if not credentials:
            raise ValueError("principal registry must not be empty")
        self._credentials = credentials

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> PrincipalAuthenticator:
        configured = (os.environ if environment is None else environment).get(
            "AXA_API_PRINCIPALS_FILE"
        )
        if not configured:
            raise ValueError("AXA_API_PRINCIPALS_FILE is required")
        return cls.from_file(configured)

    @classmethod
    def from_file(cls, path: str | Path) -> PrincipalAuthenticator:
        registry_path = Path(path)
        if not registry_path.is_absolute():
            raise ValueError("principal registry path must be absolute")
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != "axa.api_principals.v1"
        ):
            raise ValueError("unsupported principal registry schema")
        if set(payload) != {"schema_version", "principals"}:
            raise ValueError("principal registry contains unsupported fields")
        raw_principals = payload.get("principals")
        if not isinstance(raw_principals, list) or not raw_principals:
            raise ValueError("principal registry must contain principals")

        credentials: list[_PrincipalCredential] = []
        principal_ids: set[str] = set()
        credential_hashes: set[str] = set()
        for raw in raw_principals:
            if not isinstance(raw, Mapping) or set(raw) != {
                "principal_id",
                "principal_type",
                "credential_sha256",
                "roles",
            }:
                raise ValueError("principal entry contains unsupported fields")
            principal_id = str(raw["principal_id"])
            credential_hash = str(raw["credential_sha256"])
            roles = raw["roles"]
            if principal_id in principal_ids:
                raise ValueError("principal IDs must be unique")
            if not _SHA256.fullmatch(credential_hash) or credential_hash in credential_hashes:
                raise ValueError("credential hashes must be unique lowercase SHA-256 values")
            if not isinstance(roles, list) or any(
                not isinstance(role, str) or not role.strip() for role in roles
            ):
                raise ValueError("roles must contain non-empty strings")
            principal = Principal(
                principal_id=principal_id,
                principal_type=PrincipalType(raw["principal_type"]),
                roles=frozenset(roles),
            )
            credentials.append(_PrincipalCredential(principal, credential_hash))
            principal_ids.add(principal_id)
            credential_hashes.add(credential_hash)
        return cls(tuple(credentials))

    def authenticate(self, authorization: str | None) -> Principal:
        if not authorization or not authorization.startswith("Bearer "):
            raise AuthenticationError("valid bearer authentication is required")
        secret = authorization.removeprefix("Bearer ")
        if not secret or secret != secret.strip() or " " in secret:
            raise AuthenticationError("valid bearer authentication is required")
        supplied_hash = hashlib.sha256(secret.encode("utf-8")).hexdigest()
        authenticated: Principal | None = None
        for credential in self._credentials:
            if hmac.compare_digest(supplied_hash, credential.credential_sha256):
                authenticated = credential.principal
        if authenticated is None:
            raise AuthenticationError("valid bearer authentication is required")
        return authenticated

    def principal_by_id(self, principal_id: str) -> Principal:
        """Resolve a server-configured principal for internal delegation checks."""
        principal = next(
            (
                credential.principal
                for credential in self._credentials
                if credential.principal.principal_id == principal_id
            ),
            None,
        )
        if principal is None:
            raise AuthenticationError("principal is not server configured")
        return principal


class Mode(StrEnum):
    AGENT = "agent"
    JUDGE = "judge"
    RATER = "rater"
    HUMAN = "human"
    ADMIN = "admin"
    SERVICE = "service"


class Op(StrEnum):
    CREATE_CASE = "create_case"
    IMPORT = "import"
    TREE_READ = "tree_read"
    TREE_ACT = "tree_act"
    BLOB_READ = "blob_read"
    RECORD_NOTE = "record_note"
    REVIEW_READ = "review_read"
    REVIEW_WRITE = "review_write"
    REVIEW_REFIT = "review_refit"
    JUDGE_LABEL = "judge_label"
    ASSIGN = "assign"
    DECLARE_EXPOSURE = "declare_exposure"
    TRIAGE = "triage"
    EXPORT = "export"
    SCORECARD = "scorecard"


@dataclass(frozen=True)
class OpRule:
    modes: frozenset[Mode]
    grant: CaseOperation | None


RULES: dict[Op, OpRule] = {
    Op.CREATE_CASE: OpRule(frozenset({Mode.ADMIN}), None),
    Op.IMPORT: OpRule(frozenset({Mode.ADMIN, Mode.SERVICE}), None),
    Op.TREE_READ: OpRule(frozenset({Mode.AGENT, Mode.HUMAN, Mode.ADMIN}), CaseOperation.INSPECT),
    Op.TREE_ACT: OpRule(frozenset({Mode.AGENT, Mode.HUMAN, Mode.ADMIN}), CaseOperation.ACTION),
    Op.BLOB_READ: OpRule(frozenset({Mode.AGENT, Mode.HUMAN, Mode.ADMIN}), CaseOperation.ARTIFACT),
    Op.RECORD_NOTE: OpRule(frozenset({Mode.HUMAN, Mode.ADMIN}), CaseOperation.ACTION),
    Op.REVIEW_READ: OpRule(frozenset({Mode.JUDGE, Mode.RATER, Mode.ADMIN}), CaseOperation.INSPECT),
    Op.REVIEW_WRITE: OpRule(frozenset({Mode.RATER}), CaseOperation.INSPECT),
    Op.REVIEW_REFIT: OpRule(frozenset({Mode.RATER}), CaseOperation.ACTION),
    Op.JUDGE_LABEL: OpRule(frozenset({Mode.JUDGE}), CaseOperation.INSPECT),
    Op.ASSIGN: OpRule(frozenset({Mode.ADMIN}), None),
    Op.DECLARE_EXPOSURE: OpRule(frozenset({Mode.ADMIN}), None),
    Op.TRIAGE: OpRule(frozenset({Mode.ADMIN, Mode.SERVICE}), None),
    Op.EXPORT: OpRule(frozenset({Mode.ADMIN}), CaseOperation.EXPORT),
    Op.SCORECARD: OpRule(frozenset({Mode.ADMIN}), None),
}
assert set(RULES) == set(Op)


def mode_of(principal: Principal) -> Mode:
    match principal.principal_type:
        case PrincipalType.AGENT:
            return Mode.AGENT
        case PrincipalType.MODEL_JUDGE:
            return Mode.JUDGE
        case PrincipalType.SERVICE:
            return Mode.SERVICE
        case PrincipalType.USER:
            if principal.roles & LABEL_MODE_ROLES:
                if "owner" in principal.roles:
                    raise Denied()
                return Mode.RATER
            return Mode.ADMIN if "owner" in principal.roles else Mode.HUMAN
    raise Denied()


def label_kind(principal: Principal) -> str:
    mode = mode_of(principal)
    if mode is Mode.JUDGE:
        return "model_judge"
    if mode is Mode.RATER:
        return "owner" if "owner_rater" in principal.roles else "expert"
    raise AuthorizationError("no label-mode credential")


@dataclass(frozen=True)
class Access:
    principal: Principal
    root: str | None
    mode: Mode
    grants: frozenset[CaseOperation]

    @classmethod
    def for_principal(cls, principal: Principal) -> Access:
        return cls(principal, None, mode_of(principal), frozenset())

    def allows(self, op: Op) -> bool:
        rule = RULES[op]
        return self.mode in rule.modes and (rule.grant is None or rule.grant in self.grants)

    def require(self, op: Op, *, mode_only: bool = False) -> None:
        allowed = self.mode in RULES[op].modes if mode_only else self.allows(op)
        if not allowed:
            raise Denied()


def execution_environment(principal: Principal) -> str:
    return "harbor_container" if "harbor_episode" in principal.roles else "host_development"


def service_role(principal: Principal, supported: tuple[str, ...]) -> str:
    roles = [role for role in principal.roles if role in supported]
    if len(roles) != 1:
        raise AuthorizationError("a service principal holds exactly one service role")
    return roles[0]
