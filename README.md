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

AquaCommons is a sustainable fishing simulation built for the Meta PyTorch OpenEnv hackathon. It models a coastal fishing fleet operating under changing currents, weather, fuel, and quota constraints.

## Overview

This project implements an OpenEnv-compatible environment for benchmarking reinforcement learning agents on sustainable fishing decisions.

The task challenges an agent to:

- navigate a 25×25 ocean grid
- manage fuel and vessel movement
- target dense fish clusters
- deploy fishing casts while preserving quota
- react to dynamic weather, currents, and hazards

AquaCommons is designed for RL evaluation because it combines structured action and observation spaces, clear task scenarios, and reward logic that balances catch, efficiency, and sustainability.

The reward function is scaled to keep episode scores within a bounded range. Each step is scored by combining multiple numeric components:

- `catch_reward`: base 0.12 plus up to 0.45 × intensity × local_density, with higher local density and moderate cast intensity earning more points.
- `move_reward`: +0.25 for moving to a cell with at least 0.05 higher fish density, +0.12 for at least 0.02 better density, +0.05 for non-worsening movement, and -0.03 for moving into lower density.
- `efficiency_reward`: +0.02 per move on easy/medium tasks, +0.01 per move on hard tasks; +0.10 for staying on a rich location while scanning.
- `sustainability_reward`: +0.14 for responsible casts with intensity ≤ 0.65 in dense areas, -0.06 for aggressive or low-value casts, plus an extra +0.05 bonus for low-intensity casts in dense regions.
- `penalty`: +0.40 penalty for a net cast that catches nothing, +0.25 penalty for high-intensity casts in low-density water, and an automatic -0.02 step cost every step.
- `hazard_penalty`: -0.60 for entering a hazard tile in hard mode, plus extra fuel cost.
- terminal bonus: +0.80 raw reward when the agent returns safely to port with valid catch and no quota violation.

The raw reward is then normalized as `(raw_reward + 1.0) / 2.0` and clipped to the range `0.0–1.0`, producing a final per-step reward that remains comparable across episodes.

This project is built on OpenEnv and provides environment dynamics and scoring rules. The baseline `inference.py` script is a separate demo runner and is not part of the core environment API.

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/Surya-github-cloud/Aquacommon.git
cd aquacommons
```

2. Set up environment variables:

```bash
cp .env.example .env
# Edit .env and add your Hugging Face token (HF_TOKEN) from https://huggingface.co/settings/tokens
# The HF_TOKEN is required only for running the inference.py demo script
# The core OpenEnv environment (server) runs without any environment variables
```

3. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
source .venv/bin/activate # macOS/Linux
```

4. Install dependencies:

```bash
pip install -r server/requirements.txt
```

5. Run the server:

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

6. Run the demo script (requires HF_TOKEN in .env):

```bash
python inference.py
```

## Features

- OpenEnv-compliant environment with `reset()` and `step()`, plus internal state modeling
- Typed Pydantic models for actions, observations, and state
- Three benchmark tasks: `easy-calm-bay`, `medium-migrating-schools`, `hard-volatile-ocean`
- Reward logic for catch, fuel efficiency, sustainability, and hazard penalties
- Baseline demo script with step logging and final score output
- FastAPI HTTP server exposing environment endpoints
- Docker container support for local and Hugging Face Spaces deployment

## Repository Layout

- `inference.py` — REQUIRED at repository root; baseline evaluation/demo runner
- `openenv.yaml` — OpenEnv metadata, task definitions, and deployment settings
- `server/environment.py` — core OpenEnv environment implementation
- `server/app.py` — HTTP server exposing environment interfaces and health checks
- `models.py` — `AquacommonsAction`, `AquacommonsObservation`, and state schema definitions
- `server/requirements.txt` — runtime Python dependencies for the server
- `Dockerfile` — container build instructions with port handling for Spaces
- `pyproject.toml` — package metadata and installation config
- `README.md` — this submission documentation

Expected file layout:

```text
.
├── inference.py
├── openenv.yaml
├── README.md
├── Dockerfile
├── pyproject.toml
├── models.py
└── server/
    ├── app.py
    ├── environment.py
    └── requirements.txt
```

## Submission Requirements

The hackathon submission is expected to include:

- A public GitHub repository: https://github.com/Surya-github-cloud/Aquacommon
- `server/requirements.txt`
- `inference.py` located at the repository root
- `README.md`
- A deployed Hugging Face Spaces URL: https://huggingface.co/spaces/suryavamsi0818/openEnv
- A working Docker/container setup if applicable
- `openenv.yaml` and supporting environment files

Important notes for review:

- `inference.py` must be located in the root of the repository.
- Support files may exist alongside it in the repo root.
- All environment logic must run offline.
- No external APIs or cloud databases are required for the core environment.
- The submitted Hugging Face Space must be built and running.

## Prerequisites

- Python 3.10 or later
- Git for repository access and version control
- Virtual environment familiarity
- `uvicorn` for running the FastAPI server locally
- Hugging Face account with API token (for running the inference.py demo script)
- `docker` for container-based deployment or testing

> The environment itself runs offline and does not depend on external APIs. The core environment and server do not require any external API token. If `inference.py` is used for an optional LLM-based demo, that component may require an API token, but it is separate from the offline environment.

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

Then confirm startup:

```bash
curl http://localhost:8000/health
```

Run the demo script from the repository root:

```bash
python inference.py
```

## Offline Execution Requirements

- The core environment is built to run offline.
- No cloud database is required.
- No external API calls are required for the environment itself.
- Any optional LLM/demo path in `inference.py` is separate from the offline environment and is not required for validation.

## Evaluation Criteria

Reviewers can validate:

- runtime correctness of `server.environment.AquacommonsEnvironment`
- OpenEnv interface compliance (`reset()`, `step()`, state modeling)
- offline execution with no external APIs or cloud databases
- task design quality across easy, medium, and hard scenarios
- grading and reward logic that balances fishing success with sustainability and hazards
- overall code quality, project organization, and documentation

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
