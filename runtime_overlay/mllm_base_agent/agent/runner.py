"""Internal Think -> Act -> Evaluate -> Final runner.

This replaces the previous external graph state machine with a plain Python loop while
keeping the public `.invoke()` and `.stream()` shape used by legacy scripts.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from mllm_base_agent.agent.state import AgentState
from actions.response_parser import parse_vlm_response
from actions.max_steps import resolve_max_steps_from_task
from mllm_base_agent.llm.messages import AIMessage, HumanMessage, SystemMessage
from mllm_base_agent.llm.provider import get_vlm
from mllm_base_agent.prompts import get_system_prompt

LOCAL_RETRY_CONFIG = {
    'max_retries': 3,
    'api_max_retries': 5,
    'retry_delay': 2,
    'api_retry_delay': 5,
}
MODEL_HISTORY_TURNS = 29
HISTORY_CAP = 500  # per-entry cap for structured_trajectory / conversation_history


def _process_rss_gb() -> float:
    """Current python process RSS in GiB, read from /proc (no psutil dependency)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return 0.0


def _heavy_process_count() -> int:
    """Count concurrently alive Unity/Xvfb worker processes (thor, Xvfb, linux_exec)."""
    try:
        import subprocess

        out = subprocess.run(
            ["bash", "-c", "ps -eo comm= | grep -Ec '^(thor|Xvfb|linux_exec)$'"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return int(out or 0)
    except Exception:
        return 0


def _cap_history(state: AgentState, cap: int = HISTORY_CAP) -> None:
    """Bound in-memory trajectory/conversation lists; overflow is archived to disk."""
    import json as _json
    import os as _os

    run_dir = state.get("run_output_dir")
    for key in ("structured_trajectory", "conversation_history"):
        items = state.get(key) or []
        if len(items) <= cap:
            continue
        dropped = items[:-cap]
        state[key] = items[-cap:]
        if not run_dir:
            continue
        try:
            _os.makedirs(run_dir, exist_ok=True)
            with open(
                _os.path.join(run_dir, "history_archive.jsonl"),
                "a",
                encoding="utf-8",
            ) as fh:
                for item in dropped:
                    fh.write(_json.dumps(item, ensure_ascii=False) + "\n")
        except Exception:
            pass

EXTERNAL_FAILURE_TYPES = {'api_error', 'env_error', 'external_error'}
MODEL_FAILURE_TYPES = {'parse_error', 'action_error', 'model_error'}


def _object_query_cfg(state: AgentState) -> dict:
    """Opt-in WM object-query / completion-check settings (WM_QUERY=1)."""
    probe = ((state.get('config') or {}).get('memory_probe') or {})
    return probe.get('object_query') or {}


class GraphRecursionError(RuntimeError):
    """Compatibility exception for old graph error handling."""


class ParseRetryError(Exception):
    pass


class APIRetryError(Exception):
    pass


def _success_value_for_failure_type(failure_type: Optional[str]) -> Optional[bool]:
    if failure_type in EXTERNAL_FAILURE_TYPES:
        return None
    if failure_type:
        return False
    return None


def _normalize_token_usage(raw_usage: Optional[dict]) -> Dict[str, int]:
    usage = raw_usage or {}

    def to_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    prompt_tokens = to_int(usage.get('prompt_tokens'))
    completion_tokens = to_int(usage.get('completion_tokens'))
    total_tokens = to_int(usage.get('total_tokens')) or prompt_tokens + completion_tokens
    api_calls = to_int(usage.get('api_calls')) or (1 if total_tokens else 0)
    return {
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': total_tokens,
        'api_calls': api_calls,
    }


def _extract_token_usage_from_response(response: Any) -> Dict[str, int]:
    if response is None:
        return _normalize_token_usage({})
    metadata = getattr(response, 'response_metadata', None) or {}
    if isinstance(metadata, dict) and metadata.get('token_usage'):
        return _normalize_token_usage(metadata['token_usage'])
    usage_metadata = getattr(response, 'usage_metadata', None)
    if isinstance(usage_metadata, dict):
        return _normalize_token_usage(usage_metadata)
    additional_kwargs = getattr(response, 'additional_kwargs', None) or {}
    if isinstance(additional_kwargs, dict):
        return _normalize_token_usage(additional_kwargs.get('token_usage') or additional_kwargs.get('usage'))
    return _normalize_token_usage({})


def _accumulate_token_usage(state: AgentState, token_usage: Dict[str, int]) -> None:
    if 'token_usage' not in state or not isinstance(state.get('token_usage'), dict):
        state['token_usage'] = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'api_calls': 0}
    usage = state['token_usage']
    normalized = _normalize_token_usage(token_usage)
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'api_calls'):
        usage[key] = int(usage.get(key, 0) or 0) + normalized[key]


def _read_image_as_data_url(image_path: str, max_retries: int, retry_delay: int,
                            max_side: int = 0) -> str:
    """Read a frame as a data URL; optionally downscale (history frames only).

    ``max_side`` > 0 converts to JPEG with that longest side, which cuts the
    payload of long multimodal histories by ~10x.  The image of the *current*
    step is always sent untouched.
    """
    if max_side:
        try:
            import io as _io

            from PIL import Image as _Image

            with _Image.open(image_path) as im:
                im = im.convert("RGB")
                if max(im.size) > max_side:
                    scale = max_side / float(max(im.size))
                    im = im.resize((max(1, int(im.width * scale)),
                                    max(1, int(im.height * scale))),
                                   _Image.BILINEAR)
                buf = _io.BytesIO()
                im.save(buf, format="JPEG", quality=85)
            payload = base64.b64encode(buf.getvalue()).decode('utf-8')
            return f'data:image/jpeg;base64,{payload}'
        except Exception:
            pass        # fall back to the raw file below
    last_error: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            with open(image_path, 'rb') as handle:
                image_data = base64.b64encode(handle.read()).decode('utf-8')
            return f'data:image/png;base64,{image_data}'
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    raise OSError(f'Failed to read image after {max_retries} attempts: {last_error}')


def _build_messages(state: AgentState, image_url: str) -> list:
    config = state.get('config') or {}
    env_type = str(config.get('env', {}).get('type', 'ai2thor')).lower()
    context_config = config.get('context_management', {}) or {}
    enable_summary = bool(context_config.get('enable_long_term_summary', False))
    task_cfg = config.get('task', {}) or {}
    actions_cfg = config.get('actions', {}) or {}
    executor_type = state.get('executor_type')
    input_modality = task_cfg.get('input_modality') or state.get('input_modality')
    navigation_mode = actions_cfg.get('navigation_mode', 'discrete')
    vh_objects = None
    env = state.get('env')
    if env_type == 'virtualhome' and env is not None and hasattr(env, 'get_scene_interactable_object_types'):
        try:
            vh_objects = env.get_scene_interactable_object_types()
        except Exception:
            vh_objects = None
    success_criteria_block = state.get(
        'success_criteria_block',
        'Complete the task according to the instruction. Use EndTask(DONE) only after confirming success.',
    )
    prompt = get_system_prompt(
        env_type,
        enable_summary=enable_summary,
        executor_type=executor_type,
        input_modality=input_modality,
        navigation_mode=navigation_mode,
        virtualhome_interactable_object_types=vh_objects,
    ).format(
        task_prompt=state.get('task_prompt', 'Complete the task.'),
        success_criteria_block=success_criteria_block,
    )
    messages = [SystemMessage(content=prompt)]
    long_term_summary = state.get('long_term_summary', '')
    history = (state.get('short_term_history', []) or [])[-MODEL_HISTORY_TURNS:]
    for idx, entry in enumerate(history):
        content = []
        if idx == 0 and enable_summary and long_term_summary.strip():
            content.append({'type': 'text', 'text': f'**Previous Exploration Summary (Long-term Memory):**\n{long_term_summary}\n\n---\n'})
        content.append({'type': 'text', 'text': f"Step {entry.get('step', '?')}"})
        hist_image = entry.get('image_path')
        if hist_image and os.path.exists(hist_image):
            try:
                content.append({'type': 'image_url', 'image_url': {
                    'url': _read_image_as_data_url(
                        hist_image, 1, 0,
                        max_side=int(os.environ.get('WM_HISTORY_MAX_SIDE', 0) or 0))}})
            except Exception:
                content.append({'type': 'text', 'text': '[Image unavailable]'})
        messages.append(HumanMessage(content=content))
        messages.append(AIMessage(content=entry.get('raw_response', '')))
    current_content = []
    if not history and enable_summary and long_term_summary.strip():
        current_content.append({'type': 'text', 'text': f'**Previous Exploration Summary (Long-term Memory):**\n{long_term_summary}\n\n---\n'})
    goal_image_path = state.get('goal_image_path')
    if goal_image_path and state.get('step_count', 0) == 0:
        try:
            current_content.append({'type': 'text', 'text': '**Goal Image (your target destination):**'})
            current_content.append({'type': 'image_url', 'image_url': {'url': _read_image_as_data_url(goal_image_path, 1, 0)}})
        except Exception:
            pass
    current_content.append({'type': 'text', 'text': f"Step {state.get('step_count', 0)}"})
    mem_pending = state.get('_mem_pending', '')
    if mem_pending:
        current_content.append({'type': 'text', 'text': mem_pending})
        state['_mem_pending'] = ''
    # task-level plan (optional): one text call at the first step, then a
    # per-step progress line whose ticks come from the agent's action history
    plan_cfg = ((state.get('config') or {}).get('memory_probe') or {}).get('plan') or {}
    if plan_cfg.get('enabled'):
        from mllm_base_agent.agent import plan as plan_mod

        if state.get('_plan') is None:
            plan = plan_mod.decompose(
                state.get('vlm'), state.get('task_prompt', ''),
                variant=str(plan_cfg.get('prompt', 'new')))
            state['_plan'] = plan
            state['_plan_done'] = 0
            print(f"\n[Plan] {len(plan)} subgoals "
                  f"(prompt={plan_cfg.get('prompt', 'new')})", flush=True)
            for i, step in enumerate(plan, 1):
                print(f"   {i}. {step.get('goal', '')}", flush=True)
        plan_line = plan_mod.render(state.get('_plan') or [],
                                    int(state.get('_plan_done', 0) or 0))
        if plan_line:
            state['_plan_line'] = plan_line
            current_content.append({'type': 'text', 'text': plan_line})
    # opt-in: teach the query syntax once, then repeat the WM memory readout
    oq_cfg = _object_query_cfg(state)
    if oq_cfg.get('enabled'):
        from mllm_base_agent.agent import object_query as oq

        if int(state.get('step_count', 0) or 0) == 0:
            current_content.append({'type': 'text', 'text': oq.PROTOCOL_TEXT})
        oq_block = oq.render_block(state, int(state.get('step_count', 0) or 0))
        if oq_block:
            current_content.append({'type': 'text', 'text': oq_block})
    current_content.append({'type': 'image_url', 'image_url': {'url': image_url}})
    messages.append(HumanMessage(content=current_content))
    return messages


def think_node(state: AgentState) -> AgentState:
    observation = state['observation']
    vlm = state['vlm']
    state.setdefault('structured_trajectory', [])
    state.setdefault('conversation_history', [])
    state.setdefault('short_term_history', [])
    state.setdefault('long_term_summary', '')
    _cap_history(state)
    max_retries = LOCAL_RETRY_CONFIG['max_retries']
    api_max_retries = LOCAL_RETRY_CONFIG.get('api_max_retries', max_retries)
    retry_delay = LOCAL_RETRY_CONFIG['retry_delay']
    api_retry_delay = LOCAL_RETRY_CONFIG['api_retry_delay']
    config = state.get('config') or {}
    env_type = str(config.get('env', {}).get('type', 'ai2thor')).lower()
    enable_summary = bool((config.get('context_management') or {}).get('enable_long_term_summary', False))

    try:
        image_url = _read_image_as_data_url(observation.image_path, max_retries, retry_delay)
    except Exception as exc:
        state['failure_type'] = 'external_error'
        state['fail_reason'] = str(exc)
        state['should_continue'] = False
        state['success'] = None
        return state

    mem_hint_snapshot = state.get('_mem_pending', '')
    _t_build = time.time()
    messages = _build_messages(state, image_url)
    if os.environ.get('WM_STEP_TIMING'):
        _n_img = 0
        for _m in messages:
            _c = getattr(_m, 'content', None)
            if isinstance(_c, list):
                _n_img += sum(1 for _p in _c
                              if isinstance(_p, dict)
                              and _p.get('type') == 'image_url')
        print(f"[timing] 组装消息(含历史图编码) {time.time()-_t_build:.2f}s "
              f"消息数={len(messages)} 本次请求图片数={_n_img}", flush=True)
    if state.get('_plan_line'):
        mem_hint_snapshot = (mem_hint_snapshot + '\n' + state['_plan_line']).strip()
    last_error: Optional[BaseException] = None
    response_text: Optional[str] = None
    step_token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'api_calls': 0}

    for api_attempt in range(api_max_retries):
        try:
            _t_vlm = time.time()
            response = vlm.invoke(messages)
            if os.environ.get('WM_STEP_TIMING'):
                print(f"[timing] VLM 调用 {time.time()-_t_vlm:.2f}s", flush=True)
            response_text = getattr(response, 'content', str(response))
            usage = _extract_token_usage_from_response(response)
            _accumulate_token_usage(state, usage)
            for key in step_token_usage:
                step_token_usage[key] += usage.get(key, 0)
            break
        except Exception as exc:
            last_error = exc
            if api_attempt < api_max_retries - 1:
                time.sleep(api_retry_delay)
            else:
                state['failure_type'] = 'api_error'
                state['fail_reason'] = f'API error after {api_max_retries} attempts: {exc}'
                state['should_continue'] = False
                state['success'] = None
                state['structured_trajectory'].append({
                    'step': state.get('step_count', 0),
                    'thinking': '',
                    'action_string': '',
                    'action': {},
                    'raw_response': '',
                    'llm_token_usage': dict(step_token_usage),
                    'parse_error': state['fail_reason'],
                    'failure_type': 'api_error',
                    'reward': None,
                    'image_path': observation.image_path,
                })
                return state

    for parse_attempt in range(max_retries):
        try:
            oq_cfg = _object_query_cfg(state)
            if oq_cfg.get('enabled'):
                from mllm_base_agent.agent import object_query as oq

                query = oq.parse_query(oq.extract_action_string(response_text or ''))
                if query:
                    step_i = int(state.get('step_count', 0) or 0)
                    state['next_action'] = oq.apply_query(state, query, step_i)
                    state['failure_type'] = None
                    state['task_done_by_model'] = False
                    state['task_fail_by_model'] = False
                    state['structured_trajectory'].append({
                        'step': step_i,
                        'thinking': f'(object query: {query})',
                        'action_string': f'Query({query})',
                        'action': dict(state['next_action']),
                        'updated_summary': '',
                        'raw_response': (response_text or '')[:2000],
                        'llm_token_usage': dict(step_token_usage),
                        'parse_error': None,
                        'retry_count': parse_attempt,
                        'reward': None,
                        'observation_summary': None,
                        'image_path': observation.image_path,
                    })
                    state['conversation_history'].append({
                        'step': step_i,
                        'user_message': f"Step {step_i}",
                        'assistant_response': response_text or '',
                        'llm_token_usage': dict(step_token_usage),
                        'action_executed': 'CheckState',
                        'reward': None,
                        'error_message': None,
                    })
                    _shown = "CheckState()" if query == "__all__" else f"Query({query})"
                    print(f"\n[StateCheck] step {step_i}: {_shown}", flush=True)
                    return state
            parsed = parse_vlm_response(
                response_text or '',
                enable_summary=enable_summary,
                env_type=env_type,
                executor_type=state.get('executor_type'),
            )
            action = parsed['parsed_action']
            is_completion = action.get('action_type') == 'task_completion'
            state['next_action'] = action
            state['should_continue'] = not is_completion
            state['task_done_by_model'] = action.get('action_name') == 'DONE'
            state['task_fail_by_model'] = action.get('action_name') == 'FAIL'
            if enable_summary and parsed.get('updated_summary'):
                state['long_term_summary'] = parsed['updated_summary']
            trajectory_step = {
                'step': state.get('step_count', 0),
                'thinking': parsed['thinking_text'],
                'action_string': parsed['action_string'],
                'action': action,
                'updated_summary': parsed.get('updated_summary', ''),
                'raw_response': (response_text or '')[:2000],
                'llm_token_usage': dict(step_token_usage),
                'parse_error': None,
                'retry_count': parse_attempt,
                'reward': None,
                'observation_summary': None,
                'image_path': observation.image_path,
            }
            state['structured_trajectory'].append(trajectory_step)
            _append_step_log(
                state,
                {
                    'step': state.get('step_count', 0),
                    'thinking': parsed['thinking_text'],
                    'action_string': parsed['action_string'],
                    'mem_hint': mem_hint_snapshot,
                    'hidden_belief': state.get('_hidden_belief_diagnostics', {}),
                },
            )
            state['conversation_history'].append({
                'step': state.get('step_count', 0),
                'user_message': f"Step {state.get('step_count', 0)}",
                'assistant_response': response_text or '',
                'llm_token_usage': dict(step_token_usage),
                'action_executed': '',
                'reward': None,
                'error_message': None,
            })
            state['failure_type'] = None
            # NOTE: DONE is never intercepted.  The agent may end the episode
            # whenever it wants; Query(...) / Query(all) are options it can use
            # beforehand, and the choice is entirely the model's.
            return state
        except Exception as exc:
            last_error = exc
            if parse_attempt < max_retries - 1:
                try:
                    response = vlm.invoke(messages)
                    response_text = getattr(response, 'content', str(response))
                    usage = _extract_token_usage_from_response(response)
                    _accumulate_token_usage(state, usage)
                    for key in step_token_usage:
                        step_token_usage[key] += usage.get(key, 0)
                except Exception as api_exc:
                    last_error = api_exc
                time.sleep(retry_delay)

    state['failure_type'] = 'parse_error'
    state['fail_reason'] = f"Step {state.get('step_count', 0)} parse failed after {max_retries} retries: {last_error}"
    state['should_continue'] = False
    state['success'] = False
    state['structured_trajectory'].append({
        'step': state.get('step_count', 0),
        'thinking': '',
        'action_string': '',
        'action': {},
        'raw_response': (response_text or '')[:2000],
        'llm_token_usage': dict(step_token_usage),
        'parse_error': state['fail_reason'],
        'failure_type': 'parse_error',
        'reward': None,
        'image_path': observation.image_path,
    })
    return state


def _append_step_log(state: AgentState, record: dict) -> None:
    """Append one step's thinking + injected hints to steps.jsonl so that
    interrupted runs still leave a readable trace."""
    import json as _json
    import os as _os

    run_dir = state.get('run_output_dir')
    if not run_dir:
        return
    try:
        _os.makedirs(run_dir, exist_ok=True)
        with open(_os.path.join(run_dir, 'steps.jsonl'), 'a', encoding='utf-8') as fh:
            fh.write(_json.dumps(record, ensure_ascii=False) + '\n')
    except Exception:
        pass


def act_node(state: AgentState) -> AgentState:
    if state.get('think_failed') or state.get('failure_type') in {'api_error', 'parse_error', 'external_error'}:
        return state
    action = state.get('next_action')
    if not action:
        state['failure_type'] = 'action_error'
        state['fail_reason'] = 'No action available from think_node'
        state['should_continue'] = False
        state['success'] = False
        return state
    action_type = action.get('action_type')
    action_name = action.get('action_name')
    observation = state.get('observation')
    prev_image_path = getattr(observation, 'image_path', None) if observation is not None else None
    error_message = None
    # 'internal_noop' is the agent's own Query(...) / completion re-check: it
    # consumes a step but never reaches the environment.
    if action_type in ('task_completion', 'internal_noop'):
        state['step_count'] = int(state.get('step_count', 0) or 0) + 1
    else:
        try:
            observation, error_message = state['env'].step_with_action_dict(action)
            if observation is None:
                state['failure_type'] = 'action_error'
                state['fail_reason'] = f'Model output invalid action: {error_message}'
                state['should_continue'] = False
                state['success'] = False
                return state
            state['observation'] = observation
            state['step_count'] = int(state.get('step_count', 0) or 0) + 1
            if int(state['step_count']) % 10 == 0:
                print(
                    f"[mem] step {state['step_count']}: RSS={_process_rss_gb():.2f}GB "
                    f"heavy_procs={_heavy_process_count()}",
                    flush=True,
                )
            # Action outcome from the two frames the VLM was shown:
            # mse > 1 -> success, mse < 1 -> failure (every action type).
            from mllm_base_agent.agent.self_observation import (
                MOVE_ACTIONS, ROTATE_ACTIONS, estimate_success,
            )

            _ok, _mse = estimate_success(
                action.get('action_name'), prev_image_path,
                getattr(observation, 'image_path', None),
            )
            _is_move = (action.get('action_name') in MOVE_ACTIONS
                        or action.get('action_name') in ROTATE_ACTIONS)
            action_blocked = (_is_move and _ok is False)
            if action_blocked:
                print(f"\n[SelfObservation] step {state.get('step_count', 0)}: "
                      f"画面未变化（MSE={_mse:.1f}），判定该移动动作被挡", flush=True)
            mem_cfg = (state.get('config') or {}).get('memory_probe') or {}
            if mem_cfg.get('enabled'):
                from mllm_base_agent.agent.memory_probe import MemoryProbe

                probe = state.get('_mem_probe')
                if probe is None:
                    task_cfg = (state.get('config') or {}).get('task') or {}
                    probe = MemoryProbe(
                        task_description=task_cfg.get('instruction'),
                    )
                    # 目标物名字由模型自己在开局说一次（无词表、无菜单）
                    probe.set_vlm(state.get('vlm'))
                    probe.set_target_hint(mem_cfg.get('target_hint') or {})
                    state['_mem_probe'] = probe
                raw_meta = getattr(observation, 'metadata', None) or {}
                wm_cfg = mem_cfg.get('world_model') or {}
                if wm_cfg.get('enabled'):
                    from mllm_base_agent.agent.world_model import WorldModel

                    wm = state.get('_world_model')
                    if wm is None:
                        env_cfg = (state.get('config') or {}).get('env') or {}
                        actions_cfg = (state.get('config') or {}).get('actions') or {}
                        wm_kwargs = dict(
                            fov=float(env_cfg.get('field_of_view', 60)),
                            width=int(env_cfg.get('width', 800)),
                            height=int(env_cfg.get('height', 600)),
                            pose_from_action_log=bool(
                                wm_cfg.get('pose_from_action_log', True)
                            ),
                            pose_initial=wm_cfg.get('pose_initial', 'origin'),
                            move_magnitudes={
                                'MoveAhead': float(actions_cfg.get('move_ahead_magnitude', 0.5)),
                                'MoveBack': float(actions_cfg.get('move_back_magnitude', 0.5)),
                                'MoveLeft': float(actions_cfg.get('move_left_magnitude', 0.5)),
                                'MoveRight': float(actions_cfg.get('move_right_magnitude', 0.5)),
                                'MoveSmall': float(actions_cfg.get('move_small_magnitude', 0.25)),
                                'MoveMedium': float(actions_cfg.get('move_medium_magnitude', 0.5)),
                                'MoveLarge': float(actions_cfg.get('move_large_magnitude', 1.0)),
                            },
                        )
                        run_dir = state.get('run_output_dir')
                        if run_dir:
                            wm_kwargs['checkpoint_dir'] = os.path.join(
                                str(run_dir), 'world_model'
                            )
                        ckpt_path = None
                        if run_dir:
                            ckpt_path = os.path.join(
                                str(run_dir), 'world_model', 'world_model_ckpt.json'
                            )
                        resume = bool(
                            wm_cfg.get('resume')
                            or os.environ.get('WORLD_MODEL_RESUME') == '1'
                        )
                        if resume and ckpt_path and os.path.exists(ckpt_path):
                            try:
                                wm = WorldModel.load(ckpt_path, **wm_kwargs)
                                print(
                                    f"↻ Resumed world model from {ckpt_path} "
                                    f"(step {wm._step})",
                                    flush=True,
                                )
                            except Exception as exc:
                                print(
                                    f"⚠️ WorldModel checkpoint load failed: {exc}",
                                    flush=True,
                                )
                                wm = WorldModel(**wm_kwargs)
                        else:
                            wm = WorldModel(**wm_kwargs)
                        state['_world_model'] = wm
                        # V1 eyes: dense+depth runtime; simulator segmentation
                        # is disabled at runtime.
                        ckpt = (wm_cfg.get('perception_ckpt')
                                or os.environ.get('PERCEPTION_CKPT'))
                        if not ckpt and not wm_cfg.get('allow_sim_seg'):
                            raise RuntimeError(
                                "world_model enabled without perception_ckpt: "
                                "sim segmentation perception is disabled")
                        if ckpt:
                            try:
                                runtime_root = (
                                    wm_cfg.get('perception_runtime_root')
                                    or os.environ.get('LIGHTWM_ROOT')
                                    or '/home/sudidaren/lightwm_phases'
                                )
                                sys.path.insert(0, runtime_root)
                                from phase_b.runtime_factory import build_runtime
                                runtime = build_runtime({**wm_cfg, 'perception_ckpt': ckpt})
                            except Exception as exc:
                                raise RuntimeError(
                                    f"PerceptionRuntime init failed: {exc}")
                            wm.attach_perception(runtime)
                            # 注意：不再把检测器的类别表交给 MemoryProbe
                            # （基线没有词表，WM 臂也不能用；目标物名字由模型自述）
                    raw_meta = wm.observe(
                        observation,
                        action=action,
                        moved=_ok,             # frame-diff outcome (mse>1)
                        action_ok=_ok,
                        env=state.get('env'),
                    )
                if os.environ.get('WM_DEBUG'):
                    print(
                        "[WM-DEBUG] raw_meta="
                        + (str(sorted(raw_meta.keys())) if isinstance(raw_meta, dict)
                           else type(raw_meta).__name__)
                        + f" n_obj={len((raw_meta or {}).get('objects') or []) if isinstance(raw_meta, dict) else -1}"
                        + f" agent={(raw_meta or {}).get('agent') if isinstance(raw_meta, dict) else None}",
                        flush=True,
                    )
                state['_mem_pending'] = probe.update(
                    metadata=raw_meta,
                    action_name=action.get('action_name'),
                    object_type=action.get('object_type'),
                    env=state.get('env'),
                    blocked=action_blocked,   # frame diff, not error_message
                    action_ok=((raw_meta.get('action_outcome') or {}).get('ok')
                               if isinstance(raw_meta, dict) else _ok),
                    action_ok_source=(
                        (raw_meta.get('action_outcome') or {}).get('source')
                        if isinstance(raw_meta, dict) else None),
                )
                if state['_mem_pending']:
                    print(f"\n[MemoryProbe] step {state.get('step_count', 0)}: "
                          f"{state['_mem_pending'].replace(chr(10), ' | ')}", flush=True)
                oq_cfg = _object_query_cfg(state)
                if oq_cfg.get('enabled'):
                    from mllm_base_agent.agent import object_query as oq

                    mem = oq.memory_of(state)
                    mem.observe(raw_meta, int(state.get('step_count', 0) or 0))
                    mem.record_interaction(
                        int(state.get('step_count', 0) or 0),
                        action.get('action_name'),
                        action.get('object_type'),
                        ((raw_meta.get('action_outcome') or {}).get('ok')
                         if isinstance(raw_meta, dict) else _ok),
                    )
        except Exception as exc:
            state['failure_type'] = 'env_error'
            state['fail_reason'] = f'Environment exception: {exc}'
            state['should_continue'] = False
            state['success'] = None
            return state

    if state.get('structured_trajectory'):
        last_step = state['structured_trajectory'][-1]
        last_step['reward'] = 0 if action_type in ('task_completion', 'internal_noop') else getattr(observation, 'reward', 0)
        last_step['observation_summary'] = f'Task completion: {action_name}' if action_type == 'task_completion' else getattr(observation, 'text_state', '')
        last_step['error_message'] = error_message
    if state.get('conversation_history'):
        last_conv = state['conversation_history'][-1]
        last_conv['action_executed'] = action_name
        last_conv['reward'] = 0 if action_type in ('task_completion', 'internal_noop') else getattr(observation, 'reward', 0)
        last_conv['error_message'] = error_message

    # tick off plan progress from the agent's own action history
    if state.get('_plan'):
        from mllm_base_agent.agent import plan as plan_mod

        state['_plan_done'] = plan_mod.advance(
            state['_plan'], int(state.get('_plan_done', 0) or 0),
            action_name, action.get('object_type'))

    context = (state.get('config') or {}).get('context_management') or {}
    configured_history = int(context.get('short_term_history_window_size', MODEL_HISTORY_TURNS) or MODEL_HISTORY_TURNS)
    max_history = min(MODEL_HISTORY_TURNS, max(0, configured_history))
    action_string = action_name
    if action.get('object_type'):
        action_string = f"{action_name}({action.get('object_type')})"
    state.setdefault('short_term_history', []).append({
        'step': int(state.get('step_count', 1) or 1) - 1,
        'action_string': action_string,
        'reward': 0 if action_type in ('task_completion', 'internal_noop') else getattr(observation, 'reward', 0),
        'image_path': getattr(observation, 'image_path', None),
        'raw_response': state.get('structured_trajectory', [{}])[-1].get('raw_response', ''),
        'error_message': error_message,
    })
    if len(state['short_term_history']) > max_history:
        state['short_term_history'] = state['short_term_history'][-max_history:]
    _cap_history(state)
    return state


def _count_consecutive_failures(state: AgentState) -> int:
    count = 0
    for step in reversed(state.get('structured_trajectory', [])):
        reward = step.get('reward')
        if reward is None or reward < 0.05:
            count += 1
        else:
            break
    return count


def _create_env_evaluator(env_type: str, task_config: dict):
    """Create the evaluator from the environment's own implementation.

    The shared registry (``evaluation/__init__.py``) tries VirtualHome first and
    therefore returns the VirtualHome evaluator for any config with
    success_conditions, which drops environment-specific logic (e.g. CARLA's
    threshold semantics).  Dispatch explicitly to the per-environment official
    verifier; fall back to the registry for unknown env types.
    """
    import importlib

    module_map = {
        'ai2thor': 'evaluation.ai2thor.base',
        'procthor': 'evaluation.procthor.base',
        'virtualhome': 'evaluation.virtualhome.base',
        'carla': 'evaluation.carla.base',
    }
    module_name = module_map.get(str(env_type).lower())
    if module_name:
        module = importlib.import_module(module_name)
        return module.create_evaluator_from_config(task_config)
    from evaluation import create_evaluator_from_config

    return create_evaluator_from_config(task_config)


def perform_final_evaluation(state: AgentState = None, *, env=None, task_config: dict = None, observation=None) -> tuple:
    try:
        config = {}
        if state is not None:
            config = state.get('config') or {}
            task_config = config.get('task') or {}
            env = state.get('env')
            observation = state.get('observation')
        if not task_config or observation is None or not getattr(observation, 'metadata', None):
            return False, 0.0
        env_type = str((config.get('env') or {}).get('type', 'ai2thor')).lower()
        evaluator = _create_env_evaluator(env_type, task_config)
        score = evaluator.evaluate(env, observation.metadata)
        return score >= 1.0, score
    except Exception:
        return False, 0.0


def evaluate_node(state: AgentState) -> AgentState:
    if state.get('task_done_by_model'):
        success, _score = perform_final_evaluation(state)
        state['success'] = success
        state['fail_reason'] = None if success else 'Model claimed DONE but success conditions not met'
        # REMOVED 2026-09-15 (information isolation): the DONE gate used to
        # block a wrong DONE and tell the model which success conditions were
        # still unmet.  That reads (a) the task's formal success predicate from
        # task.json and (b) the simulator's object metadata -- neither of which
        # the frozen MLLM ever sees, so it was an oracle.  A wrong DONE is now
        # simply a failure, exactly as in the official baseline.
        state['should_continue'] = False
        return state
    if state.get('task_fail_by_model'):
        state['success'] = False
        state['fail_reason'] = 'Model determined task cannot be completed or refused to continue'
        state['should_continue'] = False
        return state
    if state.get('should_continue') is False:
        return state
    if int(state.get('step_count', 0) or 0) >= int(state.get('max_steps', 30) or 30):
        state['success'] = False
        state['fail_reason'] = f"Reached maximum step limit ({state.get('max_steps')} steps)"
        state['should_continue'] = False
        return state
    state['should_continue'] = True
    return state


def final_node(state: AgentState) -> AgentState:
    output_dir = state.get('run_output_dir')
    if not output_dir:
        return state
    os.makedirs(output_dir, exist_ok=True)
    env = state.get('env')
    observation = state.get('observation')
    scene_name = getattr(env, 'scene', 'UnknownScene')
    metadata = getattr(observation, 'metadata', None) or {}
    if isinstance(metadata, dict):
        scene_name = metadata.get('sceneName', scene_name)
    task_name = ((state.get('config') or {}).get('task') or {}).get('name', 'task') or 'task'
    safe_scene = str(scene_name).replace(' ', '_').replace('/', '_')[:80]
    safe_task = str(task_name).replace(' ', '_').replace('/', '_')[:80]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(output_dir, f'episode_{safe_scene}_{safe_task}_{timestamp}.json')
    episode = {
        'task': state.get('task_prompt', ''),
        'scene': scene_name,
        'success': state.get('success', False),
        'fail_reason': state.get('fail_reason'),
        'failure_type': state.get('failure_type'),
        'step_count': state.get('step_count', 0),
        'max_steps': state.get('max_steps', 0),
        'action_sequence': env.get_action_sequence() if hasattr(env, 'get_action_sequence') else '(no action records)',
        'trajectory': [
            {
                'step': item.get('step'),
                'thinking': item.get('thinking'),
                'action_string': item.get('action_string'),
                'llm_token_usage': item.get('llm_token_usage'),
                'reward': item.get('reward'),
                'error_message': item.get('error_message'),
            }
            for item in state.get('structured_trajectory', [])
        ],
        'timestamp': datetime.now().isoformat(),
        'metadata': {
            'total_reward': sum((item.get('reward') or 0) for item in state.get('structured_trajectory', [])),
            'parse_errors_count': sum(1 for item in state.get('structured_trajectory', []) if item.get('parse_error')),
            'token_usage': state.get('token_usage', {}),
        },
    }
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(episode, handle, ensure_ascii=False, indent=2)
    state['episode_log_path'] = path
    return state


class AgentRunner:
    def __init__(self, recursion_limit: int = 1000) -> None:
        self.recursion_limit = recursion_limit

    def stream(self, initial_state: AgentState, config: Optional[dict] = None) -> Iterable[Dict[str, AgentState]]:
        state = initial_state
        task_cfg = (state.get('config') or {}).get('task') or {}
        if state.get('max_steps_override') is not None:
            state['max_steps'] = int(state['max_steps_override'])
        elif task_cfg:
            state['max_steps'] = resolve_max_steps_from_task(task_cfg, int(state.get('max_steps', 30) or 30))
        limit = int((config or {}).get('recursion_limit', self.recursion_limit) or self.recursion_limit)
        iterations = 0
        while True:
            if iterations >= limit:
                raise GraphRecursionError(f'Recursion limit reached: {limit}')
            iterations += 1
            state = think_node(state)
            yield {'think': state}
            state = act_node(state)
            yield {'act': state}
            state = evaluate_node(state)
            yield {'evaluate': state}
            if not state.get('should_continue', True):
                state = final_node(state)
                yield {'final': state}
                break
        self.last_state = state

    def invoke(self, initial_state: AgentState, config: Optional[dict] = None) -> AgentState:
        final_state = initial_state
        for chunk in self.stream(initial_state, config=config):
            for update in chunk.values():
                final_state = update
        return final_state


def create_agent_graph() -> AgentRunner:
    return AgentRunner()


_parse_action_string = None
try:
    from actions.parser import parse_action_string as _parse_action_string
except Exception:
    pass

parse_action_string = _parse_action_string
_perform_final_evaluation = lambda state: perform_final_evaluation(state)[0]
execute_action = lambda env, action_dict: (*env.step_with_action_dict(action_dict), False) if action_dict.get('action_type') not in ('task_completion', 'internal_noop') else (None, None, True)
