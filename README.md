---
title: AquaCommons OpenEnv Environment
emoji: 🐟
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# AquaCommons OpenEnv Environment

AquaCommons is a sustainable fishing simulation built for the Meta PyTorch OpenEnv hackathon. It models a coastal fishing fleet operating under changing currents, weather, fuel limits, and quota constraints.

## Overview

This project implements an OpenEnv-compatible environment for benchmarking reinforcement learning agents on sustainable fishing decisions.

The task challenges an agent to:

- navigate a 25×25 ocean grid
- manage fuel and vessel movement
- target dense fish clusters
- deploy fishing casts while preserving quota
- react to dynamic weather, currents, and hazards

AquaCommons is designed for RL evaluation and grading because it combines structured action/observation spaces, scenario-based tasks, and reward logic that balances catch performance with sustainability and safety.

The reward function is scaled to keep episode scores within a bounded range. Each step is scored by combining multiple numeric components:

- `catch_reward`: base 0.12 plus up to 0.45 × intensity × local_density, with higher local density and moderate cast intensity earning more points.
- `move_reward`: +0.25 for moving to a cell with at least 0.05 higher fish density, +0.12 for at least 0.02 better density, +0.05 for non-worsening movement, and -0.03 for moving into lower density.
- `efficiency_reward`: +0.02 per move on easy/medium tasks, +0.01 per move on hard tasks; +0.10 for staying on a rich location while scanning.
- `sustainability_reward`: +0.14 for responsible casts with intensity ≤ 0.65 in dense areas, -0.06 for aggressive or low-value casts, plus an extra +0.05 bonus for low-intensity casts in dense regions.
- `penalty`: +0.40 penalty for a net cast that catches nothing, +0.25 penalty for high-intensity casts in low-density water, and an automatic -0.02 step cost every step.
- `hazard_penalty`: -0.60 for entering a hazard tile in hard mode, plus extra fuel cost.
- Terminal bonus: +0.80 raw reward when the agent returns safely to port with valid catch and no quota violation.

The raw reward is then normalized as `(raw_reward + 1.0) / 2.0` and clipped to the range `0.0–1.0`, producing a final per-step reward that remains comparable across episodes.

The baseline evaluation in `inference.py` is not a rule-based analysis system; it runs the environment and reports step-by-step actions and rewards. The project itself is built on OpenEnv and uses environment dynamics and scoring rules, not a separate handcrafted rule engine.

## Features

- OpenEnv-compliant environment with `reset()`, `step()`, and `state`
- Typed Pydantic models for actions, observations, and state
- Three benchmark tasks: `easy-calm-bay`, `medium-migrating-schools`, `hard-volatile-ocean`
- Reward logic for catch, fuel efficiency, sustainability, and hazard penalties
- Baseline inference script with step logging and final score output
- FastAPI HTTP server exposing environment endpoints
- Docker container support for local and Hugging Face Spaces deployment

## Repository Structure

- `openenv.yaml` — OpenEnv metadata, task definitions, and deployment settings
- `server/environment.py` — core OpenEnv environment implementation
- `server/app.py` — HTTP server exposing environment interfaces and health checks
- `models.py` — `AquacommonsAction`, `AquacommonsObservation`, and state schema definitions
- `inference.py` — demo and baseline execution script for task evaluation
- `server/requirements.txt` — runtime Python dependencies for the server
- `Dockerfile` — container build instructions with port handling for Spaces
- `pyproject.toml` — package metadata and installation config
- `README.md` — this submission documentation

## Prerequisites

- Python 3.10 or later
- Git for repository access and version control
- Local development familiarity with virtual environments
- `uvicorn` for running the FastAPI server locally
- `docker` for container-based deployment or testing

> The environment itself is designed to run offline and does not require cloud databases or hidden external services. The server and environment can execute locally without external APIs.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Surya-github-cloud/Aquacommon.git
cd aquacommons
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
source .venv/bin/activate # macOS/Linux
```

3. Install dependencies:

```bash
pip install -r server/requirements.txt
```

4. Confirm `uvicorn` is installed:

```bash
python -m uvicorn --help
```

## How to Run Locally

Start the OpenEnv server locally:

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Then open the local health endpoint to confirm startup:

```bash
curl http://localhost:8000/health
```

Run the demo evaluation script:

```bash
python inference.py
```

> `inference.py` is an optional baseline runner. The core environment and HTTP server do not require external APIs to run.

## Submission Contents

This submission includes:

- Public GitHub repository with source code
- `server/requirements.txt` for environment dependencies
- `Dockerfile` for container deployment
- `openenv.yaml` with OpenEnv metadata and tasks
- `inference.py` demo script for evaluation
- `README.md` documentation and usage instructions
- `server/app.py` FastAPI deployment entrypoint

If deployed to Hugging Face Spaces, the expected container runtime is the repository Docker configuration.

## Environment Constraints

- The core environment runs offline.
- No cloud database or hidden external services are required for the environment itself.
- The environment is self-contained and only depends on the Python packages listed in `server/requirements.txt`.
- Optional inference evaluation may use an API token if invoked, but is not required for OpenEnv compliance.

## Evaluation Criteria

Judges should verify:

- runtime correctness of `server.environment.AquacommonsEnvironment`
- OpenEnv interface compliance (`reset()`, `step()`, state modeling)
- task design quality across easy, medium, and hard scenarios
- grading and reward logic that balances fishing success with sustainability and hazards
- overall code quality, project organization, and documentation

## Demo / Screenshots

> Add screenshots, GIFs, or demo links here once available.

## Troubleshooting

- If dependencies fail, reinstall with:

```bash
pip install -r server/requirements.txt
```

- If the server is not accessible, verify the port and host:

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

- If Docker deployment fails, confirm the `Dockerfile` is present and build manually:

```bash
docker build -t aquacommons .
```

- If the README or healthcheck does not match, refresh the local container or restart the server.

## License / Acknowledgements

This project is submitted for the Meta PyTorch OpenEnv Hackathon.

Acknowledgements:

- OpenEnv by Meta for the environment framework
- FastAPI / Uvicorn for the HTTP runtime
- Hugging Face Spaces for deployment support
