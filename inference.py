import os
from typing import Any, Dict, List

from openai import OpenAI

from models import AquacommonsAction
from server.environment import AquacommonsEnvironment

# Environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Verify HF_TOKEN is set
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is required to run inference.py")

# Initialize OpenAI client
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# Task configuration - run single task, configurable via env var
TASK_NAME = os.getenv("TASK_NAME", "easy-calm-bay")

SYSTEM_PROMPT = (
    "You are a smart coastal fishing fleet operator. "
    "Your goal is to manage fuel, currents, weather, quota, and sustainability while harvesting fish in Indian coastal waters. "
    "For each observation, choose one action from MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST, STAY, CAST_NET, RETURN_TO_PORT. "
    "Prefer sustainable casts, move toward denser fish, conserve fuel, and return safely when quota or fuel is low. "
    "Output your action in this exact format:\n"
    "ACTION_TYPE: <action>"
)

ALLOWED_ACTIONS = {
    "MOVE_NORTH",
    "MOVE_SOUTH",
    "MOVE_EAST",
    "MOVE_WEST",
    "STAY",
    "CAST_NET",
    "RETURN_TO_PORT",
}


def format_observation(observation: Dict[str, Any]) -> str:
    """Format observation for the LLM prompt."""
    lines = []
    for key in sorted(observation.keys()):
        value = observation[key]
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def parse_action_response(text: str) -> AquacommonsAction:
    """Parse LLM response into an AquacommonsAction."""
    action_type = "STAY"
    cast_intensity = 0.0
    explanation = "No explanation provided."

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("ACTION_TYPE:"):
            action_type = line.split(":", 1)[1].strip().upper()

    # Validate action
    if action_type not in ALLOWED_ACTIONS:
        action_type = "STAY"

    return AquacommonsAction(
        action_type=action_type,
        cast_intensity=cast_intensity,
        explanation=explanation,
    )


def run_task(task_name: str) -> None:
    """Run a single task and print results in required format."""
    env = AquacommonsEnvironment()
    observation = env.reset(task=task_name)
    state = observation.model_dump()
    rewards: List[float] = []
    step_count = 0
    success = False

    print(f"[START] task={task_name} env=aquacommons model={MODEL_NAME}")

    try:
        while step_count < 60:
            step_count += 1
            error_text = "null"

            try:
                # Build prompt
                observation_str = format_observation(state)
                prompt = (
                    f"Task: {task_name}\n"
                    f"Observation:\n{observation_str}\n"
                    f"Choose one best action."
                )

                # Get action from LLM
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,  # Deterministic for evaluation
                    max_tokens=250,
                )

                response_text = response.choices[0].message.content.strip()
                action = parse_action_response(response_text)

                # Step environment
                result = env.step(action)
                reward = float(result.reward)
                done = bool(result.done)
                rewards.append(reward)

                # Print step (action string only, no extra fields)
                print(
                    f"[STEP] step={step_count} action={action.action_type} reward={reward:.2f} done={str(done).lower()} error=null"
                )

                state = result.model_dump()

                if done:
                    success = True
                    break

            except Exception as exc:
                error_text = str(exc).replace("\n", " ")
                # On error, use STAY as action and print error (fallback behavior)
                print(
                    f"[STEP] step={step_count} action=STAY reward=0.00 done=false error={error_text}"
                )
                break

    finally:
        # Always print [END]
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(success).lower()} steps={step_count} rewards={rewards_str}"
        )


def main() -> None:
    """Run the single task."""
    run_task(TASK_NAME)


if __name__ == "__main__":
    main()
