"""Generate Phase D per-task/per-group YAML configs for the AI2-THOR runner."""

from __future__ import annotations

import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from tasks import GROUPS, TASKS  # noqa: E402

OUT = "/home/sudidaren/lightwm_phases/phase_d/configs"


def make_config(task, group):
    g = GROUPS[group]
    mp = {
        "enabled": g["memory_probe"],
        "targets": task["target_object_types"],
        "navigation_directive": True,
        "interact_soft_dist": 1.2,
        "done_gate": True,
        "gate": g["gate"],
        "world_model": {
            "enabled": True,
            "pose_from_action_log": True,
            "pose_initial": "origin",
            "spatial_memory": True,
        },
    }
    cfg = {
        "env": {
            "type": "ai2thor",
            "agent_count": 1,
            "width": 800,
            "height": 600,
            "field_of_view": 60,
            "grid_size": 0.25,
            "render_depth": True,
            "render_instance_segmentation": True,
            "text_state_mode": "first_person",
            "visibility_distance": 1.0,
            "platform": None,
        },
        "max_steps": task["max_steps"],
        "context_management": {
            "enable_long_term_summary": False,
            "short_term_history_window_size": 29,
        },
        "model": {
            "vlm": {
                "provider": "openai",
                "model_name": "gpt-5",
                "temperature": 1.0,
                "max_tokens": 4096,
                "base_url": None,
                "api_key": None,
            }
        },
        "logging": {"stdout_verbose": True, "save_step_images": True},
        "actions": {
            "move_small_magnitude": 0.25,
            "move_medium_magnitude": 0.5,
            "move_large_magnitude": 1,
            "move_ahead_magnitude": 0.5,
            "move_back_magnitude": 0.5,
            "move_left_magnitude": 0.5,
            "move_right_magnitude": 0.5,
            "rotate_degrees": 90,
        },
        "reward": {
            "success_reward": 10.0,
            "step_success_bonus": 0.1,
            "step_failure_penalty": -0.05,
        },
        "experiment": {"num_episodes": 1, "output_dir": "outputs"},
        "task_presets": {
            task["id"]: {
                "scene": task["scene"],
                "description": task["description"],
                "target_object_types": task["target_object_types"],
                "success_conditions": task["success_conditions"],
                "max_steps": task["max_steps"],
            }
        },
        "memory_probe": mp,
    }
    return cfg


def main():
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for task in TASKS:
        for group in GROUPS:
            cfg = make_config(task, group)
            path = os.path.join(OUT, f"{task['id']}__{group}.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
            n += 1
    print(f"wrote {n} configs to {OUT}")


if __name__ == "__main__":
    main()
