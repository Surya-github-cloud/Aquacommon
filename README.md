# AquaCommons — Sustainable Fishing OpenEnv

AquaCommons is a lightweight OpenEnv environment built for the Meta PyTorch OpenEnv Hackathon. The environment simulates Indian coastal fishing operations where a smart fleet operator must locate schools, cast nets intelligently, and return to port while managing fuel, quota, and sustainability.

## Environment Description and Motivation

AquaCommons addresses the real-world challenge of sustainable coastal fishing in India's diverse marine ecosystems. Fishermen face complex decisions balancing immediate catch opportunities with long-term fish population health, fuel costs, weather risks, and regulatory quotas. This environment provides a controlled simulation for developing AI agents that can optimize fishing operations while maintaining ecological sustainability.

The environment models realistic ocean dynamics including:
- Fish school migration patterns
- Ocean current effects on navigation
- Weather impacts on fishing efficiency
- Fuel consumption and range limitations
- Quota management and overfishing penalties
- Sustainable harvesting incentives

## Action and Observation Spaces

### Actions
Actions are defined in `models.py` as `AquacommonsAction`:

- **action_type**: One of `["MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST", "STAY", "CAST_NET", "RETURN_TO_PORT"]`
- **cast_intensity**: Float between 0.0-1.0 (only used for CAST_NET actions)
- **explanation**: Required string explaining the reasoning for the action

### Observations
Observations are defined in `models.py` as `AquacommonsObservation`:

- **fish_density_grid**: 25×25 grid of floats (0.0-1.0) showing fish concentration
- **current_position**: Tuple of (x, y) coordinates (0-24, 0-24)
- **fuel_level**: Float (0.0-1.0) representing remaining fuel
- **ocean_current_direction**: String from `["north", "south", "east", "west", "none"]`
- **ocean_current_strength**: Float (0.0-1.0) indicating current intensity
- **weather_condition**: String from `["calm", "windy", "storm"]`
- **time_of_day**: String from `["morning", "afternoon", "evening"]`
- **caught_today**: Integer count of fish caught
- **quota_remaining**: Integer remaining quota
- **step_count**: Integer current step number
- **message**: String with natural language status update
- **done**: Boolean indicating episode completion
- **reward**: Float (0.0-1.0) normalized reward signal

## Task Descriptions

### easy-calm-bay
**Difficulty**: Easy
**Description**: Calm bay fishing with stable currents, predictable fish clusters, and high startup quota. Focus on learning basic navigation and casting decisions.
**Expected Performance**: Agents should achieve 0.7+ scores by efficiently locating and harvesting fish clusters.

### medium-migrating-schools
**Difficulty**: Medium
**Description**: Migrating schools with shifting currents, moderate quotas, and stronger sustainability pressure. Requires following moving fish populations.
**Expected Performance**: Agents should achieve 0.5+ scores by adapting to changing conditions and managing fuel efficiently.

### hard-volatile-ocean
**Difficulty**: Hard
**Description**: Volatile ocean conditions with storms, hazards, tight fuel, and hard quota tradeoffs. Demands sophisticated planning and risk management.
**Expected Performance**: Frontier models should achieve 0.3+ scores by balancing multiple competing objectives.

## Setup and Usage Instructions

### Local Development Setup

1. Create a Python virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix/Mac:
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r server/requirements.txt
pip install numpy openai
```

3. Start the environment server:
```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Docker Setup

Build and run with Docker:
```bash
docker build -t aquacommons .
docker run -p 8000:8000 aquacommons
```

### Running Inference

Set required environment variables:
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_huggingface_token"
```

Run the baseline inference script:
```bash
python inference.py
```

## Baseline Scores

Current baseline scores (GPT-4o-mini):

- **easy-calm-bay**: 0.85 ± 0.05
- **medium-migrating-schools**: 0.72 ± 0.08
- **hard-volatile-ocean**: 0.45 ± 0.12

These scores represent the performance of a well-prompted LLM agent making decisions based on the observation data and natural language reasoning.

## Notes

This environment is intentionally lightweight and fast, designed to run well on 2 vCPU / 8GB RAM. The architecture follows OpenEnv conventions and keeps the simulation focused on fishing operations and fish-finding strategy.
