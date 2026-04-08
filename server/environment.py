# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Tuple
from uuid import uuid4

import numpy as np
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import AquacommonsAction, AquacommonsObservation, AquacommonsState
except ImportError:  # pragma: no cover
    from models import AquacommonsAction, AquacommonsObservation, AquacommonsState


class AquacommonsEnvironment(Environment):
    """A realistic sustainable fishing environment for AquaCommons."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    ACTION_TYPES = [
        "MOVE_NORTH",
        "MOVE_SOUTH",
        "MOVE_EAST",
        "MOVE_WEST",
        "STAY",
        "CAST_NET",
        "RETURN_TO_PORT",
    ]

    CURRENT_DIRECTIONS = ["north", "south", "east", "west", "none"]
    WEATHER_PHASES = ["calm", "windy", "storm"]
    TIME_LABELS = ["morning", "afternoon", "evening"]
    GRID_SIZE = 25
    PORT_POSITION = (0, 0)

    TASK_CONFIG = {
        "easy-calm-bay": {
            "task_name": "easy-calm-bay",
            "difficulty": "easy",
            "label": "Calm Bay Fishing",
            "quota": 50,
            "fuel_capacity": 1.0,
            "weather_sequence": ["calm", "calm", "calm"],
            "current_directions": ["none", "east"],
            "current_strength": 0.15,
            "hazard_density": 0.0,
            "max_steps": 35,
            "cluster_centers": [((18, 18), 1.0), ((15, 6), 0.8), ((8, 14), 0.75)],
            "movement_noise": 0.03,
            "storm_chance": 0.0,
        },
        "medium-migrating-schools": {
            "task_name": "medium-migrating-schools",
            "difficulty": "medium",
            "label": "Migrating Schools & Currents",
            "quota": 35,
            "fuel_capacity": 0.9,
            "weather_sequence": ["calm", "windy", "windy"],
            "current_directions": ["north", "east", "west"],
            "current_strength": 0.38,
            "hazard_density": 0.04,
            "max_steps": 42,
            "cluster_centers": [((12, 8), 0.9), ((18, 17), 0.85), ((7, 16), 0.7)],
            "movement_noise": 0.06,
            "storm_chance": 0.12,
        },
        "hard-volatile-ocean": {
            "task_name": "hard-volatile-ocean",
            "difficulty": "hard",
            "label": "Volatile Ocean Conditions",
            "quota": 28,
            "fuel_capacity": 0.82,
            "weather_sequence": ["windy", "storm", "windy"],
            "current_directions": ["north", "south", "east", "west"],
            "current_strength": 0.62,
            "hazard_density": 0.10,
            "max_steps": 48,
            "cluster_centers": [((16, 12), 0.95), ((9, 18), 0.85), ((5, 10), 0.75)],
            "movement_noise": 0.1,
            "storm_chance": 0.25,
        },
    }

    TASK_ALIASES = {
        "easy": "easy-calm-bay",
        "medium": "medium-migrating-schools",
        "hard": "hard-volatile-ocean",
    }

    DIFFICULTY_CONFIG = {config["difficulty"]: config for config in TASK_CONFIG.values()}

    def __init__(self):
        self._seed = int(os.getenv("AQUACOMMONS_SEED", "42"))
        self._task_name = "easy-calm-bay"
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
        )
        self._grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=float)
        self._position = np.array(self.PORT_POSITION, dtype=int)
        self._fuel = 1.0
        self._caught_today = 0
        self._quota = 0
        self._sustainability_score = 1.0
        self._hazard_map = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self._overcast_history = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self._current_direction = "none"
        self._current_strength = 0.0
        self._weather_condition = "calm"
        self._time_of_day = "morning"
        self._message = ""
        self._episode_risk = 0.0

    def reset(self, task: str = "easy-calm-bay") -> AquacommonsObservation:
        self._reset_count += 1
        self._episode_id = str(uuid4())
        self._rng = np.random.default_rng(self._seed + self._reset_count)

        normalized_task = str(task or "").strip().lower()
        if normalized_task not in self.TASK_CONFIG:
            normalized_task = self.TASK_ALIASES.get(normalized_task, "easy-calm-bay")

        self._task_name = normalized_task
        self._task_config = self.TASK_CONFIG[self._task_name]
        self._difficulty_level = self._task_config["difficulty"]

        config = self._task_config
        self._grid = self._generate_fish_grid(config)
        self._position = np.array(self.PORT_POSITION, dtype=int)
        self._fuel = config["fuel_capacity"]
        self._caught_today = 0
        self._quota = config["quota"]
        self._sustainability_score = 1.0
        self._hazard_map = self._generate_hazards(config)
        self._overcast_history = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
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
        )
        self._step_count = 0
        self._message = self._build_message(
            f"Episode reset for {config['label']}. Coastal fishing task {self._task_name} has begun."
        )

        return self._build_observation(done=False, reward=0.0)

    def step(self, action: AquacommonsAction) -> AquacommonsObservation:
        self._step_count += 1
        self._state.step_count = self._step_count
        self._validate_action(action)

        previous_density = self._fish_density_at(self._position)
        previous_position = self._position.copy()
        done = False
        penalty = 0.0
        catch_reward = 0.0
        move_reward = 0.0
        efficiency_reward = 0.0
        quota_reward = 0.0
        sustainability_reward = 0.0
        storm_penalty = 0.0
        hazard_penalty = 0.0
        quota_violation = False

        if action.action_type == "STAY":
            self._fuel = max(0.0, self._fuel - 0.008)
            if previous_density > 0.45:
                efficiency_reward += 0.1
            self._message = self._build_message("The vessel holds position while scanning nearby fish density.")

        elif action.action_type == "CAST_NET":
            catch_reward, penalty, sustainability_reward = self._apply_cast(action)
            if self._caught_today > self._quota:
                quota_violation = True
                penalty += 1.0
                done = True
                self._message = self._build_message(
                    "Quota exceeded by overcatch — the fleet must stop fishing and head back to port."
                )
            else:
                self._message = self._build_message(
                    "Net cast completed. Fish response depends on local density and sustainability risk."
                )

        elif action.action_type == "RETURN_TO_PORT":
            if tuple(self._position) == self.PORT_POSITION and self._caught_today > 0:
                done = True
                self._message = self._build_message("The fleet has safely returned to port with today's catch.")
            else:
                self._move_toward_port()
                self._message = self._build_message("Returning toward port with the remaining fuel and catch.")

        else:
            self._move(action.action_type)
            new_density = self._fish_density_at(self._position)
            if new_density > previous_density + 0.05:
                move_reward += 0.25
            elif new_density > previous_density + 0.02:
                move_reward += 0.12
            elif new_density >= previous_density:
                move_reward += 0.05
            else:
                penalty += 0.03
            self._message = self._build_message("The vessel navigates to a more promising area based on ocean conditions.")

        if self._position.tolist() != previous_position.tolist():
            self._fuel = max(0.0, self._fuel - (0.02 + 0.01 * self._current_strength))
            if self._difficulty_level != "hard":
                efficiency_reward += 0.02
            else:
                efficiency_reward += 0.01

        if self._difficulty_level == "hard" and self._hazard_map[tuple(self._position)]:
            hazard_penalty += 0.6
            self._fuel = max(0.0, self._fuel - 0.05)
            self._message = self._build_message(
                "A drifting debris field was encountered, costing fuel and forcing a cautious route."
            )

        if self._fuel <= 0.0:
            done = True
            penalty += 0.9
            self._message = self._build_message(
                "Fuel exhausted at sea — the mission is over and the fleet must wait for support."
            )

        if self._step_count >= self.DIFFICULTY_CONFIG[self._difficulty_level]["max_steps"]:
            done = True
            self._message = self._build_message("The daily window has closed; fishing operations must end for the day.")

        self._update_ocean_conditions()
        self._state.total_caught = self._caught_today
        self._state.sustainability_score = float(np.clip(self._sustainability_score, 0.0, 1.0))

        raw_reward = (
            0.20
            + catch_reward
            + move_reward
            + efficiency_reward
            + quota_reward
            + sustainability_reward
            - penalty
            - storm_penalty
            - hazard_penalty
            - 0.02
        )

        if done and tuple(self._position) == self.PORT_POSITION and not quota_violation and self._caught_today > 0:
            raw_reward += 0.8

        reward = float(np.clip((raw_reward + 1.0) / 2.0, 0.0, 1.0))

        return self._build_observation(done=done, reward=reward)

    @property
    def state(self) -> State:
        return self._state

    def _generate_fish_grid(self, config: dict) -> np.ndarray:
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=float)
        for (x_center, y_center), intensity in config["cluster_centers"]:
            radius = 4 if self._difficulty_level == "easy" else 5
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    x = x_center + dx
                    y = y_center + dy
                    if 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE:
                        distance = np.hypot(dx, dy)
                        attenuation = np.exp(-(distance**2) / (2.5 * 2.5))
                        grid[y, x] += intensity * attenuation
        grid += self._rng.random((self.GRID_SIZE, self.GRID_SIZE)) * 0.04
        return np.clip(grid, 0.0, 1.0)

    def _generate_hazards(self, config: dict) -> np.ndarray:
        hazard_count = int(self.GRID_SIZE * self.GRID_SIZE * config["hazard_density"])
        map_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        for _ in range(hazard_count):
            x = int(self._rng.integers(0, self.GRID_SIZE))
            y = int(self._rng.integers(0, self.GRID_SIZE))
            if (x, y) != self.PORT_POSITION:
                map_grid[y, x] = True
        return map_grid

    def _fish_density_at(self, position: np.ndarray) -> float:
        x, y = int(position[0]), int(position[1])
        return float(self._grid[y, x])

    def _validate_action(self, action: AquacommonsAction) -> None:
        if action.action_type not in self.ACTION_TYPES:
            raise ValueError(f"Invalid action_type: {action.action_type}")
        if not action.explanation or not action.explanation.strip():
            raise ValueError("Action explanation is required and cannot be empty.")

    def _apply_cast(self, action: AquacommonsAction) -> Tuple[float, float, float]:
        intensity = float(np.clip(action.cast_intensity, 0.0, 1.0))
        x, y = int(self._position[0]), int(self._position[1])
        local_density = self._fish_density_at(self._position)
        self._fuel = max(0.0, self._fuel - 0.015 - 0.005 * intensity)
        self._overcast_history[y, x] += 1

        if local_density < 0.18 and intensity > 0.4:
            return 0.0, 0.5, -0.1

        catch_strength = intensity * local_density
        catch_units = int(np.floor(catch_strength * 11.0 + 0.5))
        catch_units = max(0, catch_units)
        self._caught_today += catch_units

        density_loss = intensity * 0.18 + self._overcast_history[y, x] * 0.03
        self._grid[y, x] = max(0.0, self._grid[y, x] - density_loss)
        self._sustainability_score -= 0.03 * self._overcast_history[y, x]

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

    def _move(self, action_type: str) -> None:
        if action_type == "MOVE_NORTH":
            self._position[1] = max(0, self._position[1] - 1)
        elif action_type == "MOVE_SOUTH":
            self._position[1] = min(self.GRID_SIZE - 1, self._position[1] + 1)
        elif action_type == "MOVE_EAST":
            self._position[0] = min(self.GRID_SIZE - 1, self._position[0] + 1)
        elif action_type == "MOVE_WEST":
            self._position[0] = max(0, self._position[0] - 1)

    def _move_toward_port(self) -> None:
        if self._position[0] > self.PORT_POSITION[0]:
            self._position[0] -= 1
        elif self._position[0] < self.PORT_POSITION[0]:
            self._position[0] += 1
        elif self._position[1] > self.PORT_POSITION[1]:
            self._position[1] -= 1
        elif self._position[1] < self.PORT_POSITION[1]:
            self._position[1] += 1
        self._fuel = max(0.0, self._fuel - 0.03)

    def _update_ocean_conditions(self) -> None:
        config = self.DIFFICULTY_CONFIG[self._difficulty_level]
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

    def _shift_fish_with_current(self, config: dict) -> None:
        drift = self._rng.normal(0.0, config["movement_noise"], size=(self.GRID_SIZE, self.GRID_SIZE))
        drift = np.clip(drift, -0.04, 0.04)
        shifted = self._shift_grid(self._grid, self._current_direction, int(round(self._current_strength * 2)))
        scatter = 0.06 if self._weather_condition == "storm" else 0.02
        self._grid = np.clip(self._grid * (1.0 - scatter) + shifted * scatter + drift, 0.0, 1.0)

    def _shift_grid(self, grid: np.ndarray, direction: str, step: int) -> np.ndarray:
        if direction == "none" or step == 0:
            return grid.copy()
        shifted = np.zeros_like(grid)
        if direction == "north":
            shifted[:-step, :] = grid[step:, :]
        elif direction == "south":
            shifted[step:, :] = grid[:-step, :]
        elif direction == "east":
            shifted[:, step:] = grid[:, :-step]
        elif direction == "west":
            shifted[:, :-step] = grid[:, step:]
        return shifted

    def _build_message(self, summary: str) -> str:
        return (
            f"{summary} Current {self._weather_condition} weather, "
            f"current {self._current_direction} at strength {self._current_strength:.2f}. "
            f"Fuel {self._fuel:.2f}, quota {max(0, self._quota - self._caught_today)} remaining."
        )

    def _build_observation(self, done: bool, reward: float) -> AquacommonsObservation:
        return AquacommonsObservation(
            fish_density_grid=np.round(self._grid, 3).tolist(),
            current_position=(int(self._position[0]), int(self._position[1])),
            fuel_level=float(np.clip(self._fuel, 0.0, 1.0)),
            ocean_current_direction=self._current_direction,
            ocean_current_strength=float(np.clip(self._current_strength, 0.0, 1.0)),
            weather_condition=self._weather_condition,
            time_of_day=self._time_of_day,
            caught_today=int(self._caught_today),
            quota_remaining=int(max(0, self._quota - self._caught_today)),
            step_count=int(self._step_count),
            message=self._message,
            done=done,
            reward=float(np.clip(reward, 0.0, 1.0)),
        )
