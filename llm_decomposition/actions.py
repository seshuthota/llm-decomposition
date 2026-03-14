from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any


@dataclass(slots=True)
class ActionRecord:
    action_id: str
    action_type: str
    target_granularity: str
    target_name: str
    byte_cost: int
    proxy_family: str
    proxy_score: float
    predicted_gain_per_byte: float | None = None
    base_run_id: str | None = None
    bit_from: int | None = None
    bit_to: int | None = None
    rank: int | None = None
    rank_from: int | None = None
    rank_to: int | None = None
    rank_delta: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    selected: bool = False
    selection_order: int | None = None
    cumulative_budget_bytes: int | None = None
    status: str = "pending"

    def with_selection(self, selection_order: int, cumulative_budget_bytes: int) -> "ActionRecord":
        return replace(
            self,
            selected=True,
            selection_order=selection_order,
            cumulative_budget_bytes=cumulative_budget_bytes,
            status="selected",
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}
