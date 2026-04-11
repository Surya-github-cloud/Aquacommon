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

The environment is designed for RL evaluation with structured action and observation spaces, clear task scenarios, and reward logic that balances fishing success, efficiency, and sustainability.

The reward function encourages:
- **Efficient fishing**: Rewards for catching fish in dense areas with moderate effort
- **Fuel conservation**: Bonuses for smart navigation with minimal waste
- **Sustainable practices**: Incentives for responsible casting and quota compliance
- **Risk management**: Penalties for hazards, wasted casts, and quota violations

Rewards are normalized to `[0.0, 1.0]` for consistent comparison across episodes. The baseline `inference.py` script is a separate demo runner and not part of the core environment.

## Benchmark Tasks

AquaCommons includes five tasks inspired by real-world Indian ocean governance challenges, each with unique environmental conditions and challenges:

### Easy: Single Zone MSP (`easy-msp-single-zone`)
- **Conditions**: Stable currents, predictable fish clusters, calm weather, single vessel
- **Challenges**: High startup quota allows for exploration, but requires efficient navigation to dense areas
- **Goal**: Learn basic movement and casting strategies in a forgiving environment
- **Real-world mapping**: Individual fisher in traditional coastal fishing zones

### Medium: Multi-Agent Basic Negotiation (`medium-msp-multi-agent-basic-negotiation`)
- **Conditions**: Shifting currents, moderate quotas, 3 vessels with basic negotiation
- **Challenges**: Vessels negotiate fishing zones, balance individual vs collective interests
- **Goal**: Learn cooperative zone allocation and conflict avoidance
- **Real-world mapping**: Small fishing cooperatives negotiating territorial waters

### Hard: Full Stochastic Conflict Resolution (`hard-msp-full-stochastic-conflict-resolution`)
- **Conditions**: Storms, hazards, tight fuel, 5 vessels with full stochasticity
- **Challenges**: Random hazards, climate shocks, quota violations, complex conflict resolution
- **Goal**: Master risk management, emergency responses, and sustainable harvesting under pressure
- **Real-world mapping**: Large fishing fleets in contested waters with environmental uncertainty

### Policy: Policy Experimentation Mode (`policy-experimentation-mode`)
- **Conditions**: Government sets rules (MPA size, carbon price, quotas), 4 vessels react
- **Challenges**: Agents adapt to policy changes, balance economic incentives with conservation
- **Goal**: Maximize collective ocean health + economy through policy design
- **Real-world mapping**: India's Sagarmala/PMMSY/Deep Ocean Mission policy frameworks

### Climate: Climate Shock Resilience (`climate-shock-resilience`)
- **Conditions**: 8+ vessels + random extreme events (cyclone, marine heatwave, illegal mining)
- **Challenges**: Sudden environmental shocks, adaptive fleet management, recovery strategies
- **Goal**: Build resilience to climate change impacts and anthropogenic disturbances
- **Real-world mapping**: Climate-vulnerable Indian coastal communities facing cyclones and habitat loss

## Real-World Mapping & Blue Carbon

AquaCommons draws inspiration from India's maritime initiatives:

- **Sagarmala Initiative**: Port-led development and coastal economic zones
- **PMMSY (Pradhan Mantri Matsya Sampada Yojana)**: Blue economy and sustainable fisheries
- **Deep Ocean Mission**: Exploration and conservation of deep-sea resources
- **Blue Carbon Credit Marketplace**: Agents earn rewards for protecting/restoring seagrass beds and deploying artificial reefs
- **Emergent Metrics**: Cooperation index (vessel coordination), equity score (Gini coefficient of catches), ocean health index (biomass + carbon sequestration)

The environment simulates dynamic ocean conditions with PyTorch-accelerated currents and biomass diffusion, providing lightweight but realistic marine spatial planning challenges.

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/Surya-github-cloud/Aquacommon.git
cd aquacommons
```

2. (Optional) Set up environment variables for the demo script:

If you want to run the `inference.py` demo script with LLM-based features, copy the template and add your Hugging Face token:

```bash
copy .env.example .env  # Windows
cp .env.example .env    # macOS/Linux
```

Then edit `.env` and add your Hugging Face token from https://huggingface.co/settings/tokens

> **Note**: The core environment requires no environment variables and can be tested without this step.
> The `inference.py` demo script is optional and can run with or without an HF token (LLM features are disabled without it).

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

6. Run a specific task with the demo script:

```bash
python inference.py easy-calm-bay
python inference.py medium-migrating-schools
python inference.py hard-volatile-ocean
```

If no task is provided, the script defaults to `easy-calm-bay`. The demo script is optional for evaluation; judges can also test the environment directly via the HTTP server.

## Features

- OpenEnv-compatible environment with `reset()` and `step()` methods
- Typed Pydantic models for actions and observations
- Three benchmark tasks with increasing difficulty
- Comprehensive reward system balancing catch, efficiency, and sustainability
- Baseline demo script with step logging and score tracking
- FastAPI HTTP server with environment endpoints
- Docker support for deployment to Hugging Face Spaces

## Repository Layout

**Core Environment**
- `server/environment.py` — OpenEnv-compatible environment implementation
- `server/app.py` — HTTP/WebSocket server for environment interaction
- `models.py` — Pydantic models for actions and observations
- `server/requirements.txt` — runtime dependencies

**Demo & Configuration**
- `inference.py` — Optional demo script at repository root (separate from core)
- `openenv.yaml` — Environment metadata and task definitions
- `.env.example` — Template for optional environment variables

**Deployment**
- `Dockerfile` — Container image for Hugging Face Spaces
- `pyproject.toml` — Package configuration
- `README.md` — This file

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



## Hackathon Submission

**GitHub**: https://github.com/Surya-github-cloud/Aquacommon  
**Deployed Space**: https://huggingface.co/spaces/suryavamsi0818/openEnv (public, running)

### Core Requirements Met
✓ Public GitHub repository with full source code  
✓ `inference.py` at repository root (optional demo, separate from core)  
✓ `server/requirements.txt` with all dependencies  
✓ `README.md` documentation  
✓ Deployed, publicly accessible Hugging Face Space  
✓ OpenEnv-compatible environment: `reset()` and `step()` methods  
✓ Zero external API dependencies for core simulation  
✓ Three benchmark tasks with consistent scoring  

**Team Note**: If you qualified as a solo participant in Round 1, you will be paired into a team for the finale.

## Testing Locally

**Quick setup for judges or development:**

```bash
git clone https://github.com/Surya-github-cloud/Aquacommon.git
cd aquacommons
python -m venv .venv
.venv\Scripts\activate  # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r server/requirements.txt
python inference.py easy-calm-bay
```

The demo script will print structured output showing each step and episode score. No setup or credentials are required for this.

Alternatively, judges can interact with the deployed Space directly at https://huggingface.co/spaces/suryavamsi0818/openEnv

## Troubleshooting

- **Import errors**: Ensure dependencies are installed: `pip install -r server/requirements.txt`
- **HF Space not loading**: The Space is deployed at https://huggingface.co/spaces/suryavamsi0818/openEnv and should be live and public
- **Demo script issues**: Try a fresh virtual environment: `python -m venv .venv && source .venv/bin/activate && pip install -r server/requirements.txt`

---

**Submitted for**: Meta PyTorch OpenEnv Hackathon

**Built with**: OpenEnv, FastAPI, Hugging Face Spaces
