# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Tuple
from uuid import uuid4

import numpy as np
import torch
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import AquacommonsAction, AquacommonsObservation, AquacommonsState
except ImportError:  # pragma: no cover
    from models import AquacommonsAction, AquacommonsObservation, AquacommonsState


class AquacommonsEnvironment(Environment):
    """A realistic sustainable fishing environment for AquaCommons with multi-agent and policy features."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    ACTION_TYPES = [
        "MOVE_NORTH",
        "MOVE_SOUTH",
        "MOVE_EAST",
        "MOVE_WEST",
        "STAY",
        "CAST_NET",
        "RETURN_TO_PORT",
        "SET_POLICY",
        "NEGOTIATE",
    ]

    CURRENT_DIRECTIONS = ["north", "south", "east", "west", "none"]
    WEATHER_PHASES = ["calm", "windy", "storm"]
    TIME_LABELS = ["morning", "afternoon", "evening"]
    GRID_SIZE = 25
    PORT_POSITION = (0, 0)
    MAX_AGENTS = 10

    TASK_CONFIG = {
        "easy-msp-single-zone": {
            "task_name": "easy-msp-single-zone",
            "difficulty": "easy",
            "label": "Single Zone MSP",
            "num_agents": 1,
            "quota": 50,
            "fuel_capacity": 1.0,
            "weather_sequence": ["calm"],
            "current_directions": ["none"],
            "current_strength": 0.1,
            "hazard_density": 0.0,
            "max_steps": 35,
            "cluster_centers": [((18, 18), 1.0)],
            "movement_noise": 0.03,
            "storm_chance": 0.0,
            "blue_carbon_enabled": False,
            "climate_shocks": False,
            "policy_mode": False,
            "negotiation_mode": False,
        },
        "medium-msp-multi-agent-basic-negotiation": {
            "task_name": "medium-msp-multi-agent-basic-negotiation",
            "difficulty": "medium",
            "label": "Multi-Agent Basic Negotiation",
            "num_agents": 3,
            "quota": 35,
            "fuel_capacity": 0.9,
            "weather_sequence": ["calm", "windy"],
            "current_directions": ["north", "east"],
            "current_strength": 0.3,
            "hazard_density": 0.04,
            "max_steps": 42,
            "cluster_centers": [((12, 8), 0.9), ((18, 17), 0.85)],
            "movement_noise": 0.06,
            "storm_chance": 0.1,
            "blue_carbon_enabled": True,
            "climate_shocks": False,
            "policy_mode": False,
            "negotiation_mode": True,
        },
        "hard-msp-full-stochastic-conflict-resolution": {
            "task_name": "hard-msp-full-stochastic-conflict-resolution",
            "difficulty": "hard",
            "label": "Full Stochastic Conflict Resolution",
            "num_agents": 5,
            "quota": 28,
            "fuel_capacity": 0.8,
            "weather_sequence": ["windy", "storm"],
            "current_directions": ["north", "south", "east", "west"],
            "current_strength": 0.5,
            "hazard_density": 0.1,
            "max_steps": 48,
            "cluster_centers": [((16, 12), 0.95), ((9, 18), 0.85)],
            "movement_noise": 0.1,
            "storm_chance": 0.2,
            "blue_carbon_enabled": True,
            "climate_shocks": True,
            "policy_mode": False,
            "negotiation_mode": True,
        },
        "policy-experimentation-mode": {
            "task_name": "policy-experimentation-mode",
            "difficulty": "policy",
            "label": "Policy Experimentation Mode",
            "num_agents": 4,
            "quota": 40,
            "fuel_capacity": 1.0,
            "weather_sequence": ["calm", "windy"],
            "current_directions": ["east"],
            "current_strength": 0.2,
            "hazard_density": 0.02,
            "max_steps": 50,
            "cluster_centers": [((10, 10), 0.8), ((20, 20), 0.8)],
            "movement_noise": 0.05,
            "storm_chance": 0.05,
            "blue_carbon_enabled": True,
            "climate_shocks": False,
            "policy_mode": True,
            "negotiation_mode": False,
        },
        "climate-shock-resilience": {
            "task_name": "climate-shock-resilience",
            "difficulty": "climate",
            "label": "Climate Shock Resilience",
            "num_agents": 8,
            "quota": 30,
            "fuel_capacity": 0.7,
            "weather_sequence": ["windy", "storm", "calm"],
            "current_directions": ["north", "south", "east", "west"],
            "current_strength": 0.6,
            "hazard_density": 0.15,
            "max_steps": 60,
            "cluster_centers": [((5, 5), 0.9), ((15, 15), 0.9), ((20, 5), 0.9)],
            "movement_noise": 0.12,
            "storm_chance": 0.3,
            "blue_carbon_enabled": True,
            "climate_shocks": True,
            "policy_mode": False,
            "negotiation_mode": False,
        },
    }

    TASK_ALIASES = {
        "easy": "easy-msp-single-zone",
        "medium": "medium-msp-multi-agent-basic-negotiation",
        "hard": "hard-msp-full-stochastic-conflict-resolution",
    }

    DIFFICULTY_CONFIG = {config["difficulty"]: config for config in TASK_CONFIG.values()}

    def __init__(self):
        self._seed = int(os.getenv("AQUACOMMONS_SEED", "42"))
        self._task_name = "easy-msp-single-zone"
        self._task_config = self.TASK_CONFIG[self._task_name]
        self._difficulty_level = self._task_config["difficulty"]
        self._rng = np.random.default_rng(self._seed)
        self._reset_count = 0
        self._episode_id = str(uuid4())
        self._state = AquacommonsState(
            episode_id=self._episode_id,
            total_caught=0,
            sustainability_score=1.0,
            difficulty_level=self._difficulty_level,
            step_count=0,
            cooperation_index=0.0,
            equity_score=0.0,
            ocean_health_index=0.0,
        )
        self._grid = torch.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=torch.float32)
        self._blue_carbon_grid = torch.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=torch.float32)
        self._position = torch.tensor(self.PORT_POSITION, dtype=torch.int32)
        self._fuel = 1.0
        self._caught_today = 0
        self._quota = 0
        self._sustainability_score = 1.0
        self._hazard_map = torch.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=torch.bool)
        self._overcast_history = torch.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=torch.int32)
        self._current_direction = "none"
        self._current_strength = 0.0
        self._weather_condition = "calm"
        self._time_of_day = "morning"
        self._message = ""
        self._episode_risk = 0.0
        self._num_agents = 1
        self._vessel_positions = [self._position.clone() for _ in range(self.MAX_AGENTS)]
        self._vessel_fuels = [1.0] * self.MAX_AGENTS
        self._vessel_catches = [0] * self.MAX_AGENTS
        self._vessel_quotas = [0] * self.MAX_AGENTS
        self._policy_mpa_size = 0.0
        self._policy_carbon_price = 0.0
        self._cooperation_index = 0.0
        self._equity_score = 0.0
        self._ocean_health_index = 0.0
        self._negotiation_state = {}

    def reset(self, task: str = "easy-msp-single-zone") -> AquacommonsObservation:
        self._reset_count += 1
        self._episode_id = str(uuid4())
        self._rng = np.random.default_rng(self._seed + self._reset_count)

        normalized_task = str(task or "").strip().lower()
        if normalized_task not in self.TASK_CONFIG:
            normalized_task = self.TASK_ALIASES.get(normalized_task, "easy-msp-single-zone")

        self._task_name = normalized_task
        self._task_config = self.TASK_CONFIG[self._task_name]
        self._difficulty_level = self._task_config["difficulty"]
        self._num_agents = self._task_config["num_agents"]

        config = self._task_config
        self._grid = self._generate_fish_grid(config)
        self._blue_carbon_grid = torch.zeros_like(self._grid)
        self._position = torch.tensor(self.PORT_POSITION, dtype=torch.int32)
        self._fuel = config["fuel_capacity"]
        self._caught_today = 0
        self._quota = config["quota"]
        self._sustainability_score = 1.0
        self._hazard_map = self._generate_hazards(config)
        self._overcast_history = torch.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=torch.int32)
        self._current_direction = self._rng.choice(config["current_directions"])
        self._current_strength = config["current_strength"]
        self._weather_condition = config["weather_sequence"][0]
        self._time_of_day = "morning"
        self._state = AquacommonsState(
            episode_id=self._episode_id,
            total_caught=0,
            sustainability_score=self._sustainability_score,
            difficulty_level=self._difficulty_level,
            step_count=0,
            cooperation_index=0.0,
            equity_score=0.0,
            ocean_health_index=self._calculate_ocean_health(),
        )
        self._step_count = 0
        self._episode_risk = 0.0
        self._vessel_positions = [torch.tensor(self.PORT_POSITION, dtype=torch.int32) for _ in range(self._num_agents)]
        self._vessel_fuels = [config["fuel_capacity"]] * self._num_agents
        self._vessel_catches = [0] * self._num_agents
        self._vessel_quotas = [config["quota"]] * self._num_agents
        self._policy_mpa_size = 0.0
        self._policy_carbon_price = 0.0
        self._cooperation_index = 0.0
        self._equity_score = 0.0
        self._ocean_health_index = self._calculate_ocean_health()
        self._negotiation_state = {}
        self._message = self._build_message(
            f"Episode reset for {config['label']}. MSP task {self._task_name} has begun."
        )

        return self._build_observation(done=False, reward=0.0)

    def step(self, action: AquacommonsAction) -> AquacommonsObservation:
        self._step_count += 1
        self._state.step_count = self._step_count
        self._validate_action(action)

        done = False
        reward = 0.0

        if self._task_config["policy_mode"]:
            self._apply_policy(action)
            self._simulate_agents()
        elif self._task_config["negotiation_mode"]:
            self._apply_negotiation(action)
            self._simulate_agents()
        else:
            # Single agent mode
            reward = self._apply_single_action(action)

        # Update ocean conditions
        self._update_ocean_conditions()

        # Calculate metrics
        self._calculate_metrics()

        # Check termination
        if self._check_termination():
            done = True
            reward += self._calculate_final_reward()

        # Update state
        self._state.total_caught = sum(self._vessel_catches[:self._num_agents])
        self._state.sustainability_score = float(np.clip(self._sustainability_score, 0.0, 1.0))
        self._state.cooperation_index = self._cooperation_index
        self._state.equity_score = self._equity_score
        self._state.ocean_health_index = self._ocean_health_index

        return self._build_observation(done=done, reward=reward)

    def _apply_single_action(self, action: AquacommonsAction) -> float:
        agent_id = 0
        previous_density = self._fish_density_at(self._vessel_positions[agent_id])
        previous_position = self._vessel_positions[agent_id].clone()
        reward = 0.0
        penalty = 0.0
        catch_reward = 0.0
        move_reward = 0.0
        efficiency_reward = 0.0
        sustainability_reward = 0.0
        hazard_penalty = 0.0
        quota_violation = False

        if action.action_type == "STAY":
            self._vessel_fuels[agent_id] = max(0.0, self._vessel_fuels[agent_id] - 0.008)
            if previous_density > 0.45:
                efficiency_reward += 0.1
            self._message = self._build_message("The vessel holds position while scanning nearby fish density.")

        elif action.action_type == "CAST_NET":
            catch_reward, penalty, sustainability_reward = self._apply_cast(agent_id, action)
            if self._vessel_catches[agent_id] > self._vessel_quotas[agent_id]:
                quota_violation = True
                penalty += 1.0
                self._message = self._build_message(
                    "Quota exceeded by overcatch — the fleet must stop fishing and head back to port."
                )
            else:
                self._message = self._build_message(
                    "Net cast completed. Fish response depends on local density and sustainability risk."
                )

        elif action.action_type == "RETURN_TO_PORT":
            if tuple(self._vessel_positions[agent_id].tolist()) == self.PORT_POSITION and self._vessel_catches[agent_id] > 0:
                self._message = self._build_message("The fleet has safely returned to port with today's catch.")
            else:
                self._move_toward_port(agent_id)
                self._message = self._build_message("Returning toward port with the remaining fuel and catch.")

        else:
            self._move(agent_id, action.action_type)
            new_density = self._fish_density_at(self._vessel_positions[agent_id])
            if new_density > previous_density + 0.05:
                move_reward += 0.25
            elif new_density > previous_density + 0.02:
                move_reward += 0.12
            elif new_density >= previous_density:
                move_reward += 0.05
            else:
                penalty += 0.03
            self._message = self._build_message("The vessel navigates to a more promising area based on ocean conditions.")

        if not torch.equal(self._vessel_positions[agent_id], previous_position):
            self._vessel_fuels[agent_id] = max(0.0, self._vessel_fuels[agent_id] - (0.02 + 0.01 * self._current_strength))
            if self._difficulty_level != "hard":
                efficiency_reward += 0.02
            else:
                efficiency_reward += 0.01

        if self._task_config["blue_carbon_enabled"] and self._rng.random() < 0.1:
            # Deploy blue carbon
            x, y = self._vessel_positions[agent_id].tolist()
            self._blue_carbon_grid[y, x] = min(1.0, self._blue_carbon_grid[y, x] + 0.1)
            reward += 0.05  # Blue carbon bonus

        if self._hazard_map[tuple(self._vessel_positions[agent_id].tolist())]:
            hazard_penalty += 0.6
            self._vessel_fuels[agent_id] = max(0.0, self._vessel_fuels[agent_id] - 0.05)
            self._message = self._build_message(
                "A drifting debris field was encountered, costing fuel and forcing a cautious route."
            )

        raw_reward = (
            0.20
            + catch_reward
            + move_reward
            + efficiency_reward
            + sustainability_reward
            - penalty
            - hazard_penalty
            - 0.02
        )

        if quota_violation:
            raw_reward -= 1.0

        return float(np.clip((raw_reward + 1.0) / 2.0, 0.0, 1.0))

    def _apply_policy(self, action: AquacommonsAction):
        self._policy_mpa_size = action.policy_mpa_size
        self._policy_carbon_price = action.policy_carbon_price
        # Ensure quotas list has correct length
        if action.policy_quotas and len(action.policy_quotas) >= self._num_agents:
            self._vessel_quotas = action.policy_quotas[:self._num_agents]
        else:
            # Use default quota from config for all agents
            default_quota = self._task_config["quota"]
            self._vessel_quotas = [default_quota] * self._num_agents
        # MPA: restrict areas
        mpa_mask = torch.rand((self.GRID_SIZE, self.GRID_SIZE)) < self._policy_mpa_size
        self._grid = torch.where(mpa_mask, self._grid * 0.5, self._grid)
        # Carbon price affects fuel cost
        self._fuel_cost_multiplier = 1.0 + self._policy_carbon_price

    def _apply_negotiation(self, action: AquacommonsAction):
        self._negotiation_state['offer'] = action.negotiation_offer
        # Simple negotiation: if offer contains 'zone', adjust quotas
        if 'zone' in action.negotiation_offer.lower():
            for i in range(self._num_agents):
                self._vessel_quotas[i] = int(self._vessel_quotas[i] * 0.9)

    def _simulate_agents(self):
        for i in range(self._num_agents):
            if i == 0 and not self._task_config["policy_mode"]:
                continue  # RL agent
            # Simple simulation
            if self._rng.random() < 0.5:
                direction = self._rng.choice(["MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST"])
                self._move(i, direction)
            else:
                self._apply_cast(i, AquacommonsAction(action_type="CAST_NET", cast_intensity=0.5, explanation="Simulated"))

    def _calculate_metrics(self):
        catches = torch.tensor(self._vessel_catches[:self._num_agents], dtype=torch.float32)
        if self._num_agents > 1:
            mean_catch = catches.mean()
            self._equity_score = 1.0 - (torch.abs(catches - mean_catch).sum() / (2 * self._num_agents * mean_catch)) if mean_catch > 0 else 0.0
            # Cooperation: inverse of conflicts
            conflicts = 0
            for i in range(self._num_agents):
                for j in range(i+1, self._num_agents):
                    if torch.norm(self._vessel_positions[i].float() - self._vessel_positions[j].float()) < 2:
                        conflicts += 1
            self._cooperation_index = 1.0 - (conflicts / (self._num_agents * (self._num_agents - 1) / 2))
        else:
            self._equity_score = 1.0
            self._cooperation_index = 1.0
        self._ocean_health_index = self._grid.mean().item() + self._blue_carbon_grid.mean().item() * 0.1

    def _calculate_ocean_health(self) -> float:
        return self._grid.mean().item() + self._blue_carbon_grid.mean().item() * 0.1

    def _calculate_final_reward(self) -> float:
        score = 0.0
        if self._task_config["policy_mode"]:
            score = (self._ocean_health_index + self._equity_score + self._cooperation_index) / 3.0
        else:
            total_catch = sum(self._vessel_catches[:self._num_agents])
            score = min(1.0, total_catch / (self._num_agents * self._task_config["quota"]))
            score = score * 0.5 + self._ocean_health_index * 0.3 + self._equity_score * 0.2
        return score

    def _check_termination(self) -> bool:
        if self._step_count >= self._task_config["max_steps"]:
            return True
        for i in range(self._num_agents):
            if self._vessel_fuels[i] <= 0.0 or self._vessel_catches[i] > self._vessel_quotas[i]:
                return True
        return False

    @property
    def state(self) -> State:
        """Return the current state of the environment."""
        return self._state

    def _generate_fish_grid(self, config: dict) -> torch.Tensor:
        grid = torch.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=torch.float32)
        for (x_center, y_center), intensity in config["cluster_centers"]:
            radius = 4 if self._difficulty_level == "easy" else 5
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    x = x_center + dx
                    y = y_center + dy
                    if 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE:
                        distance = torch.hypot(torch.tensor(dx, dtype=torch.float32), torch.tensor(dy, dtype=torch.float32))
                        attenuation = torch.exp(-(distance**2) / (2.5 * 2.5))
                        grid[y, x] += intensity * attenuation
        grid += torch.rand((self.GRID_SIZE, self.GRID_SIZE)) * 0.04
        return torch.clamp(grid, 0.0, 1.0)

    def _generate_hazards(self, config: dict) -> torch.Tensor:
        hazard_count = int(self.GRID_SIZE * self.GRID_SIZE * config["hazard_density"])
        map_grid = torch.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=torch.bool)
        for _ in range(hazard_count):
            x = int(self._rng.integers(0, self.GRID_SIZE))
            y = int(self._rng.integers(0, self.GRID_SIZE))
            if (x, y) != self.PORT_POSITION:
                map_grid[y, x] = True
        return map_grid

    def _fish_density_at(self, position: torch.Tensor) -> float:
        x, y = int(position[0].item()), int(position[1].item())
        return float(self._grid[y, x])

    def _validate_action(self, action: AquacommonsAction) -> None:
        if action.action_type not in self.ACTION_TYPES:
            raise ValueError(f"Invalid action_type: {action.action_type}")
        if not action.explanation or not action.explanation.strip():
            raise ValueError("Action explanation is required and cannot be empty.")

    def _apply_cast(self, agent_id: int, action: AquacommonsAction) -> Tuple[float, float, float]:
        intensity = float(np.clip(action.cast_intensity, 0.0, 1.0))
        pos = self._vessel_positions[agent_id]
        x, y = int(pos[0].item()), int(pos[1].item())
        local_density = self._fish_density_at(pos)
        fuel_cost = 0.015 + 0.005 * intensity
        if hasattr(self, '_fuel_cost_multiplier'):
            fuel_cost *= self._fuel_cost_multiplier
        self._vessel_fuels[agent_id] = max(0.0, self._vessel_fuels[agent_id] - fuel_cost)
        self._overcast_history[y, x] = self._overcast_history[y, x] + 1

        if local_density < 0.18 and intensity > 0.4:
            return 0.0, 0.5, -0.1

        catch_strength = intensity * local_density
        catch_units = int(np.floor(catch_strength * 11.0 + 0.5))
        catch_units = max(0, catch_units)
        self._vessel_catches[agent_id] += catch_units

        density_loss = intensity * 0.18 + self._overcast_history[y, x].item() * 0.03
        self._grid[y, x] = max(0.0, self._grid[y, x] - density_loss)
        self._sustainability_score -= 0.03 * self._overcast_history[y, x].item()

        collect_factor = min(local_density, 0.9)
        catch_reward = 0.12 + collect_factor * intensity * 0.45
        sustainability_reward = 0.14 if intensity <= 0.65 and local_density >= 0.35 else -0.06
        penalty = 0.0

        if catch_units == 0:
            penalty += 0.4
        if intensity > 0.8 and local_density < 0.45:
            penalty += 0.25
        if intensity <= 0.5 and local_density >= 0.45:
            sustainability_reward += 0.05

        return catch_reward, penalty, sustainability_reward

    def _move(self, agent_id: int, action_type: str) -> None:
        pos = self._vessel_positions[agent_id]
        if action_type == "MOVE_NORTH":
            pos[1] = max(0, pos[1] - 1)
        elif action_type == "MOVE_SOUTH":
            pos[1] = min(self.GRID_SIZE - 1, pos[1] + 1)
        elif action_type == "MOVE_EAST":
            pos[0] = min(self.GRID_SIZE - 1, pos[0] + 1)
        elif action_type == "MOVE_WEST":
            pos[0] = max(0, pos[0] - 1)

    def _move_toward_port(self, agent_id: int) -> None:
        pos = self._vessel_positions[agent_id]
        if pos[0] > self.PORT_POSITION[0]:
            pos[0] -= 1
        elif pos[0] < self.PORT_POSITION[0]:
            pos[0] += 1
        elif pos[1] > self.PORT_POSITION[1]:
            pos[1] -= 1
        elif pos[1] < self.PORT_POSITION[1]:
            pos[1] += 1
        fuel_cost = 0.03
        if hasattr(self, '_fuel_cost_multiplier'):
            fuel_cost *= self._fuel_cost_multiplier
        self._vessel_fuels[agent_id] = max(0.0, self._vessel_fuels[agent_id] - fuel_cost)

    def _update_ocean_conditions(self) -> None:
        config = self._task_config
        self._time_of_day = self.TIME_LABELS[(self._step_count // 8) % len(self.TIME_LABELS)]

        if self._difficulty_level != "easy" and self._step_count % 5 == 0:
            self._weather_condition = self._rng.choice(config["weather_sequence"])
        elif self._difficulty_level == "easy":
            self._weather_condition = config["weather_sequence"][0]

        if self._difficulty_level == "hard" and self._rng.random() < config["storm_chance"]:
            self._weather_condition = "storm"

        if self._difficulty_level == "hard" and self._rng.random() < 0.2:
            self._current_direction = self._rng.choice(self.CURRENT_DIRECTIONS)

        if self._difficulty_level == "medium" and self._step_count % 7 == 0:
            self._current_direction = self._rng.choice(config["current_directions"])

        if self._difficulty_level == "easy" and self._step_count % 9 == 0:
            self._current_direction = self._rng.choice(config["current_directions"])

        self._current_strength = float(np.clip(config["current_strength"] + self._rng.normal(0.0, 0.06), 0.0, 1.0))
        self._shift_fish_with_current(config)

        # Climate shock
        if config["climate_shocks"] and self._rng.random() < 0.1:
            self._apply_climate_shock()

    def _shift_fish_with_current(self, config: dict) -> None:
        drift = torch.randn((self.GRID_SIZE, self.GRID_SIZE)) * config["movement_noise"]
        drift = torch.clamp(drift, -0.04, 0.04)
        shifted = self._shift_grid(self._grid, self._current_direction, int(round(self._current_strength * 2)))
        scatter = 0.06 if self._weather_condition == "storm" else 0.02
        self._grid = torch.clamp(self._grid * (1.0 - scatter) + shifted * scatter + drift, 0.0, 1.0)

    def _shift_grid(self, grid: torch.Tensor, direction: str, step: int) -> torch.Tensor:
        if direction == "none" or step == 0:
            return grid.clone()
        shifted = torch.zeros_like(grid)
        if direction == "north":
            shifted[:-step, :] = grid[step:, :]
        elif direction == "south":
            shifted[step:, :] = grid[:-step, :]
        elif direction == "east":
            shifted[:, step:] = grid[:, :-step]
        elif direction == "west":
            shifted[:, :-step] = grid[:, step:]
        return shifted

    def _apply_climate_shock(self) -> None:
        shock_type = self._rng.choice(["cyclone", "heatwave", "mining"])
        if shock_type == "cyclone":
            self._grid *= 0.8
        elif shock_type == "heatwave":
            self._grid *= 0.9
        elif shock_type == "mining":
            x = self._rng.integers(0, self.GRID_SIZE)
            y = self._rng.integers(0, self.GRID_SIZE)
            self._grid[y-2:y+3, x-2:x+3] *= 0.5

    def _build_message(self, summary: str) -> str:
        agent_id = 0
        return (
            f"{summary} Current {self._weather_condition} weather, "
            f"current {self._current_direction} at strength {self._current_strength:.2f}. "
            f"Fuel {self._vessel_fuels[agent_id]:.2f}, quota {max(0, self._vessel_quotas[agent_id] - self._vessel_catches[agent_id])} remaining."
        )

    def _build_observation(self, done: bool, reward: float) -> AquacommonsObservation:
        return AquacommonsObservation(
            fish_density_grid=self._grid.numpy().round(3).tolist(),
            current_position=tuple(self._vessel_positions[0].tolist()),
            fuel_level=float(np.clip(self._vessel_fuels[0], 0.0, 1.0)),
            ocean_current_direction=self._current_direction,
            ocean_current_strength=float(np.clip(self._current_strength, 0.0, 1.0)),
            weather_condition=self._weather_condition,
            time_of_day=self._time_of_day,
            caught_today=int(self._vessel_catches[0]),
            quota_remaining=int(max(0, self._vessel_quotas[0] - self._vessel_catches[0])),
            step_count=int(self._step_count),
            message=self._message,
            cooperation_index=self._cooperation_index,
            equity_score=self._equity_score,
            ocean_health_index=self._ocean_health_index,
            vessel_positions=[tuple(p.tolist()) for p in self._vessel_positions[:self._num_agents]],
            vessel_fuels=self._vessel_fuels[:self._num_agents],
            vessel_catches=self._vessel_catches[:self._num_agents],
            done=done,
            reward=float(np.clip(reward, 0.0, 1.0)),
        )
