import asyncio
import os
import re
from typing import Any, Dict, List, Optional

from models import AquacommonsAction
from server.environment import AquacommonsEnvironment

TASKS = [
    "easy-calm-bay",
    "medium-migrating-schools",
    "hard-volatile-ocean",
]
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")  # Required for Hugging Face authentication

SYSTEM_PROMPT = (
    "You are a smart coastal fishing fleet operator. "
    "Your goal is to manage fuel, currents, weather, quota, and sustainability while harvesting fish in Indian coastal waters. "
    "For each observation, choose one action from MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST, STAY, CAST_NET, RETURN_TO_PORT. "
    "Prefer sustainable casts, move toward denser fish, conserve fuel, and return safely when quota or fuel is low. "
    "Output only the requested action fields in plain text, one per line: ACTION_TYPE, CAST_INTENSITY, EXPLANATION."
)

STEP_PROMPT_TEMPLATE = (
    "Task: {task}\n"
    "Observation:\n{observation}\n"
    "Choose one best action. "
    "Use ACTION_TYPE, CAST_INTENSITY, and EXPLANATION exactly as described. "
    "If the fleet is already in a good spot, STAY is acceptable."
)

ACTION_PATTERN = re.compile(r"^(ACTION_TYPE|CAST_INTENSITY|EXPLANATION)\s*:\s*(.*)$", re.IGNORECASE)
ALLOWED_ACTIONS = {
    "MOVE_NORTH",
    "MOVE_SOUTH",
    "MOVE_EAST",
    "MOVE_WEST",
    "STAY",
    "CAST_NET",
    "RETURN_TO_PORT",
}


def build_prompt(observation: Dict[str, Any], task: str) -> str:
    observation_text = json_safe_observation(observation)
    return STEP_PROMPT_TEMPLATE.format(task=task, observation=observation_text)


def json_safe_observation(observation: Dict[str, Any]) -> str:
    lines: List[str] = []
    for key in sorted(observation.keys()):
        value = observation[key]
        if isinstance(value, list):
            lines.append(f"{key}: {value}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def parse_action_response(text: str) -> AquacommonsAction:
    action_type: Optional[str] = None
    cast_intensity: float = 0.0
    explanation: str = "No explanation provided."

    for line in text.splitlines():
        match = ACTION_PATTERN.match(line.strip())
        if not match:
            continue
        key = match.group(1).upper()
        value = match.group(2).strip()
        if key == "ACTION_TYPE":
            action_type = value.upper()
        elif key == "CAST_INTENSITY":
            try:
                cast_intensity = float(value)
            except ValueError:
                cast_intensity = 0.0
        elif key == "EXPLANATION":
            explanation = value

    if action_type not in ALLOWED_ACTIONS:
        action_type = "STAY"
    cast_intensity = float(max(0.0, min(cast_intensity, 1.0)))
    return AquacommonsAction(action_type=action_type, cast_intensity=cast_intensity, explanation=explanation)


async def choose_action(observation: Dict[str, Any], task_name: str, step_number: int) -> AquacommonsAction:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required to run inference.py")

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    prompt = build_prompt(observation, task_name)
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
        max_tokens=250,
    )

    text = response.choices[0].message.content.strip()
    action = parse_action_response(text)
    return action


async def run_task(task_name: str) -> None:
    env = AquacommonsEnvironment()
    observation = env.reset(task=task_name)
    state = observation.model_dump()
    rewards: List[float] = []
    step_count = 0
    success = False
    error_text = "null"

    print(f"[START] task={task_name} env=aquacommons model={MODEL_NAME}")
    while step_count < 60:
        step_count += 1
        try:
            action = await choose_action(state, task_name, step_count)
            result = env.step(action)
            reward = float(result.reward)
            done = bool(result.done)
            rewards.append(reward)
            print(
                f"[STEP] step={step_count} action={action.action_type} cast_intensity={action.cast_intensity:.2f} "
                f"reward={reward:.2f} done={str(done).lower()} error={error_text}"
            )
            state = result.model_dump()
            if done:
                success = True
                break
        except Exception as exc:
            error_text = str(exc).replace("\n", " ")
            print(
                f"[STEP] step={step_count} action=ERROR cast_intensity=0.00 reward=0.00 done=false error={error_text}"
            )
            break

    score = sum(rewards)
    rewards_string = ",".join(f"{r:.3f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={step_count} score={score:.3f} rewards={rewards_string}"
    )


async def main() -> None:
    for task_name in TASKS:
        await run_task(task_name)


if __name__ == "__main__":
    import json

    asyncio.run(main())
