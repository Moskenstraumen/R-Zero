"""Thin wrapper to register a custom rollout hook for slime."""

from __future__ import annotations

from typing import Any

from slime.rollout.sglang_rollout import generate_rollout as _generate_rollout
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput


def generate_rollout(
    args, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    return _generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
