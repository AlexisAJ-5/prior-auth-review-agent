"""Payer medical policies: loading, validation and the criteria logic tree."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, model_validator

from .schemas import PlanId

# A logic node is a criterion id, or {"all_of": [...]} / {"any_of": [...]} of nodes.
LogicNode = str | dict[str, list[Any]]


class Criterion(BaseModel):
    id: str
    text: str


class PolicySection(BaseModel):
    section_id: str
    heading: str
    text: str


class Policy(BaseModel):
    policy_id: str
    title: str
    service_code: str
    service_name: str
    version: str
    effective_date: str
    synthetic: bool
    applies_to_plans: list[PlanId]
    supported_diagnoses: dict[str, str]
    criteria: list[Criterion]
    logic: LogicNode
    sections: list[PolicySection]

    @model_validator(mode="after")
    def _logic_references_known_criteria(self) -> "Policy":
        known = {c.id for c in self.criteria}
        used = set(logic_criteria(self.logic))
        if used != known:
            raise ValueError(f"{self.policy_id}: logic uses {sorted(used)} but criteria are {sorted(known)}")
        return self

    def criterion(self, criterion_id: str) -> Criterion | None:
        return next((c for c in self.criteria if c.id == criterion_id), None)

    def describe_logic(self) -> str:
        return describe_logic(self.logic)


def logic_criteria(node: LogicNode) -> list[str]:
    if isinstance(node, str):
        return [node]
    ((op, children),) = node.items()
    if op not in ("all_of", "any_of"):
        raise ValueError(f"Unknown logic operator '{op}'")
    return [cid for child in children for cid in logic_criteria(child)]


def describe_logic(node: LogicNode) -> str:
    """Human-readable form, e.g. 'RF1 OR (C1 AND C2 AND C3)'."""
    if isinstance(node, str):
        return node
    ((op, children),) = node.items()
    joiner = " AND " if op == "all_of" else " OR "
    parts = [describe_logic(c) if isinstance(c, str) else f"({describe_logic(c)})" for c in children]
    return joiner.join(parts)


class PolicyRepository:
    def __init__(self, policies: list[Policy]):
        self._by_service = {p.service_code: p for p in policies}
        if len(self._by_service) != len(policies):
            raise ValueError("Two policies share a service_code")

    @classmethod
    def from_directory(cls, directory: Path) -> "PolicyRepository":
        files = sorted(directory.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No policy files in {directory}")
        return cls([Policy.model_validate(json.loads(f.read_text(encoding="utf-8"))) for f in files])

    def get(self, service_code: str) -> Policy | None:
        return self._by_service.get(service_code)

    def all(self) -> list[Policy]:
        return list(self._by_service.values())
