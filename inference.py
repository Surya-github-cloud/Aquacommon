import argparse
import os
import numpy as np
from typing import Any, Dict, List

# Load environment variables from .env file if available (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback: manually load a local .env file if python-dotenv is not installed.
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(dotenv_path):
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)

from openai import OpenAI

from models import AquacommonsAction
from server.environment import AquacommonsEnvironment

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is required to run inference.py")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

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
    lines = []
    for key in sorted(observation.keys()):
        value = observation[key]
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def safe_model_dump(obj) -> Dict[str, Any]:
    """Safely extract dict from observation/result, handling Pydantic or plain dict."""
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            return {}
    if isinstance(obj, dict):
        return obj
    try:
        return dict(obj)
    except Exception:
        return {}


def parse_action_response(text: str) -> AquacommonsAction:
    action_type = "STAY"
    cast_intensity = 0.0
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("ACTION_TYPE:"):
            action_type = line.split(":", 1)[1].strip().upper()
        elif line.startswith("CAST_INTENSITY:"):
            try:
                cast_intensity = float(line.split(":", 1)[1].strip())
            except Exception:
                cast_intensity = 0.0
    if action_type not in ALLOWED_ACTIONS:
        action_type = "STAY"
    return AquacommonsAction(
        action_type=action_type,
        cast_intensity=float(np.clip(cast_intensity, 0.0, 1.0)),
        explanation="LLM selected action",
    )


def run_task(task_name: str) -> None:
    rewards: List[float] = []
    step_count = 0
    success = False
    env = None

    print(f"[START] task={task_name} env=aquacommons model={MODEL_NAME}")

    try:
        env = AquacommonsEnvironment()
        observation = env.reset(task=task_name)
        state = safe_model_dump(observation)

        while step_count < 60:
            current_step = step_count + 1
            try:
                observation_str = format_observation(state)
                prompt = (
                    f"Task: {task_name}\n"
                    f"Observation:\n{observation_str}\n"
                    f"Choose one best action."
                )

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=250,
                )

                response_text = response.choices[0].message.content.strip()
                action = parse_action_response(response_text)

                # Simple retry if action is fallback (likely invalid)
                if action.action_type == "STAY" and "ACTION_TYPE:" not in response_text:
                    retry_prompt = prompt + "\nPlease respond with a valid ACTION_TYPE from the allowed actions."
                    retry_response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": retry_prompt},
                        ],
                        temperature=0.0,
                        max_tokens=250,
                    )
                    retry_text = retry_response.choices[0].message.content.strip()
                    action = parse_action_response(retry_text)

                # Validate action before stepping
                if action.action_type not in ALLOWED_ACTIONS:
                    action = AquacommonsAction(action_type="STAY", cast_intensity=0.0, explanation="Fallback action")

                result = env.step(action)
                reward = float(result.reward)
                done = bool(result.done)
                rewards.append(reward)
                step_count += 1  # Increment after successful step

                print(
                    f"[STEP] step={current_step} action={action.action_type} cast_intensity={action.cast_intensity:.2f} reward={reward:.2f} done={str(done).lower()} error=null"
                )

                state = safe_model_dump(result)

                if done:
                    success = True
                    break

            except Exception as exc:
                error_text = str(exc).replace("\n", " ").replace("\r", " ")
                print(
                    f"[STEP] step={current_step} action=STAY cast_intensity=0.00 reward=0.00 done=false error={error_text}"
                )
                break

    except Exception as exc:
        error_text = str(exc).replace("\n", " ").replace("\r", " ")
        print(
            f"[STEP] step=0 action=STAY cast_intensity=0.00 reward=0.00 done=false error={error_text}"
        )

    finally:
        if env and hasattr(env, 'close'):
            env.close()
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        score = sum(rewards)
        print(
            f"[END] success={str(success).lower()} steps={step_count} rewards={rewards_str}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Aquacommons inference for a specified task."
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=os.getenv("TASK_NAME", "easy-calm-bay"),
        help="Task name to run (defaults to TASK_NAME in .env or easy-calm-bay)",
    )
    args = parser.parse_args()
    run_task(args.task)


if __name__ == "__main__":
    main()
