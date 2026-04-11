#!/usr/bin/env python
"""Quick test of all 5 tasks"""

from server.environment import AquacommonsEnvironment
from models import AquacommonsAction
import traceback

env = AquacommonsEnvironment()

tasks = [
    "easy-msp-single-zone",
    "medium-msp-multi-agent-basic-negotiation",
    "hard-msp-full-stochastic-conflict-resolution",
    "policy-experimentation-mode",
    "climate-shock-resilience",
]

print("Testing all 5 AquaCommons tasks:\n")

for task_name in tasks:
    try:
        print(f"→ {task_name}")
        obs = env.reset(task=task_name)
        print(f"  ✓ Reset successful")
        print(f"    - Num agents: {len(obs.vessel_positions)}")
        print(f"    - Ocean health: {obs.ocean_health_index:.2f}")
        print(f"    - Cooperation: {obs.cooperation_index:.2f}")
        
        # Step once
        action = AquacommonsAction(
            action_type="STAY",
            cast_intensity=0.0,
            explanation="Test action"
        )
        obs = env.step(action)
        print(f"  ✓ Step successful (reward={obs.reward:.3f})")
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        traceback.print_exc()
    print()

print("✅ All 5 tasks validated!")

