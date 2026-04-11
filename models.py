# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Literal, Tuple

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, confloat, conint


class AquacommonsAction(Action):
    """Action describing a coastal fishing fleet operator decision."""

    action_type: Literal[
        "MOVE_NORTH",
        "MOVE_SOUTH",
        "MOVE_EAST",
        "MOVE_WEST",
        "STAY",
        "CAST_NET",
        "RETURN_TO_PORT",
        "SET_POLICY",
        "NEGOTIATE",
    ] = Field(..., description="Navigation and fishing command for the vessel")
    cast_intensity: confloat(ge=0.0, le=1.0) = Field(
        0.0,
        description="How much net effort to use; higher intensity increases catch and sustainability risk.",
    )
    explanation: str = Field(..., description="Human-readable reason for the chosen action")
    policy_mpa_size: float = Field(0.0, description="Marine Protected Area size for policy tasks")
    policy_carbon_price: float = Field(0.0, description="Carbon price for policy tasks")
    policy_quotas: List[int] = Field(default_factory=list, description="Quotas for each vessel")
    negotiation_offer: str = Field("", description="Negotiation offer for multi-agent tasks")


class AquacommonsObservation(Observation):
    """Observation containing ocean state, vessel status, and quota tracking."""

    fish_density_grid: List[List[float]] = Field(
        ..., description="25×25 grid showing current fish concentration across the fishing area"
    )
    current_position: Tuple[int, int] = Field(..., description="Current vessel position as (x, y)")
    fuel_level: confloat(ge=0.0, le=1.0) = Field(..., description="Remaining fuel normalized from 0.0 to 1.0")
    ocean_current_direction: Literal["north", "south", "east", "west", "none"] = Field(
        ..., description="Dominant ocean current direction"
    )
    ocean_current_strength: confloat(ge=0.0, le=1.0) = Field(
        ..., description="Relative strength of the ocean current"
    )
    weather_condition: Literal["calm", "windy", "storm"] = Field(
        ..., description="Current weather affecting fishing and visibility"
    )
    time_of_day: Literal["morning", "afternoon", "evening"] = Field(
        ..., description="Time of day in the episode"
    )
    caught_today: conint(ge=0) = Field(..., description="Total fish caught so far today")
    quota_remaining: conint(ge=0) = Field(..., description="Remaining daily quota")
    step_count: conint(ge=0) = Field(..., description="Current step count for the episode")
    message: str = Field(..., description="Natural-language ocean status update")
    cooperation_index: float = Field(..., description="Measure of cooperation between agents")
    equity_score: float = Field(..., description="Equity score (Gini coefficient)")
    ocean_health_index: float = Field(..., description="Overall ocean health index")
    vessel_positions: List[Tuple[int, int]] = Field(default_factory=list, description="Positions of all vessels")
    vessel_fuels: List[float] = Field(default_factory=list, description="Fuel levels of all vessels")
    vessel_catches: List[int] = Field(default_factory=list, description="Catches of all vessels")


class AquacommonsState(State):
    """Hidden environment state used by the grader and server."""

    episode_id: str = Field(..., description="Unique episode identifier")
    total_caught: int = Field(..., description="Cumulative catch for the episode")
    sustainability_score: float = Field(..., description="Hidden sustainability score used for grading")
    difficulty_level: Literal["easy", "medium", "hard", "policy", "climate"] = Field(
        ..., description="Selected difficulty level for the episode"
    )
    step_count: int = Field(..., description="Number of steps taken so far")
    cooperation_index: float = Field(..., description="Cooperation index")
    equity_score: float = Field(..., description="Equity score")
    ocean_health_index: float = Field(..., description="Ocean health index")
