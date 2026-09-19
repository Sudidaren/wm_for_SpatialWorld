"""LLM/VLM providers without external chain frameworks dependencies."""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Any, Dict, List, Optional, Sequence

import requests
from dotenv import load_dotenv

from mllm_base_agent.llm.messages import ModelResponse, to_openai_messages

load_dotenv()

#: 2026-09-19：这台工作站到网关的 IPv6 出网是坏的（SYN-SENT 卡死），而
#: Python 的 requests 不像 curl 会做双栈竞速——它只取 getaddrinfo 的第一条，
#: 于是每个请求都要等满超时再重试，整批任务停摆。`VLM_FORCE_IPV4=1` 时把
#: 解析强制到 IPv4，绕开这个问题（默认关闭，不影响能正常走 IPv6 的机器）。
if os.environ.get("VLM_FORCE_IPV4") == "1":
    _orig_getaddrinfo = socket.getaddrinfo

    def _getaddrinfo_v4(host, port, family=0, type=0, proto=0, flags=0):  # type: ignore[no-untyped-def]
        return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = _getaddrinfo_v4  # type: ignore[assignment]

GEMINI_RESPONSES_MODELS = {
    "Gemini 3-Pro-Preview",
    "Gemini-3-Flash-Preview",
    "Gemini-3.1-Pro-Preview",
}

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


API_RETRY_CONFIG = {
    "max_retries": _env_int("VLM_API_MAX_RETRIES", 5),
    "retry_delay": _env_int("VLM_API_RETRY_DELAY", 2),
    "retry_delay_400": _env_int("VLM_API_RETRY_DELAY_400", 5),
    "retry_delay_500": _env_int("VLM_API_RETRY_DELAY_500", 3),
    "exponential_backoff": True,
}


def _normalize_usage_dict(result: Dict[str, Any]) -> Dict[str, int]:
    usage = result.get("usage") or result.get("usage_metadata") or {}
    if not isinstance(usage, dict):
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        else:
            usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
    if "prompt_tokens" not in usage:
        usage["prompt_tokens"] = usage.get("promptTokenCount") or usage.get("prompt_token_count")
    if "completion_tokens" not in usage:
        usage["completion_tokens"] = usage.get("candidatesTokenCount") or usage.get("completion_token_count")
    if "total_tokens" not in usage:
        usage["total_tokens"] = usage.get("totalTokenCount") or usage.get("total_token_count")

    def to_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    prompt_tokens = to_int(usage.get("prompt_tokens"))
    completion_tokens = to_int(usage.get("completion_tokens"))
    total_tokens = to_int(usage.get("total_tokens")) or prompt_tokens + completion_tokens
    normalized = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    return {k: v for k, v in normalized.items() if v}


def _extract_content(result: Dict[str, Any]) -> str:
    choices = result.get("choices") or []
    if not choices:
        raise ValueError(f"Unexpected API response without choices: {result}")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, dict):
        return str(content.get("text", content))
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content or "")


class OpenAICompatibleChatModel:
    """Small `.invoke(messages)` model for OpenAI-compatible chat APIs."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: Optional[float] = None,
        timeout: int = 300,
        client_max_retries: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not api_key:
            raise ValueError("API key is required for the selected provider")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.client_max_retries = client_max_retries
        self.model_kwargs = model_kwargs or {}
        self._client = None

    @property
    def chat_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _get_openai_client(self) -> Any:
        if self._client is not None:
            return self._client or None
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url.rstrip("/") + "/",
                timeout=self.timeout,
                max_retries=0,
            )
        except Exception:
            self._client = False
        return self._client or None

    def _request_with_openai_client(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        client = self._get_openai_client()
        if client is None:
            raise RuntimeError("openai client unavailable")
        response = client.chat.completions.create(**payload)
        choices = []
        for choice in getattr(response, "choices", []) or []:
            msg = getattr(choice, "message", None)
            if msg is not None:
                choices.append({"message": {"content": getattr(msg, "content", "")}})
        usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        return {
            "choices": choices,
            "usage": usage or {},
            "model": getattr(response, "model", payload.get("model")),
        }

    def _request_with_requests(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        body = json.dumps(payload)
        response = requests.post(self.chat_url, headers=headers, data=body, timeout=self.timeout)
        if response.status_code >= 400:
            body = response.text.strip()
            if len(body) > 1000:
                body = body[:1000] + "..."
            raise requests.HTTPError(
                f"{response.status_code} Error for url: {response.url}; body: {body}",
                response=response,
            )
        return response.json()

    def _make_api_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # 调试用：把真实请求体落盘（run 目录的 log.json 是从历史轨迹重拼的，
        # 不含运行时注入的块，只有这里能证明"注入到底有没有上线"）。
        _dump = os.environ.get("VLM_DUMP_DIR")
        if _dump:
            try:
                os.makedirs(_dump, exist_ok=True)
                with open(os.path.join(_dump, f"req_{int(time.time() * 1000)}.json"),
                          "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(payload, ensure_ascii=False))
            except Exception:
                pass
        max_retries = (
            self.client_max_retries
            if self.client_max_retries is not None
            else API_RETRY_CONFIG["max_retries"]
        )
        last_error: Optional[BaseException] = None
        for attempt in range(max_retries):
            try:
                try:
                    result = self._request_with_openai_client(payload)
                except Exception:
                    result = self._request_with_requests(payload)
                if not result.get("choices"):
                    raise ValueError("API returned empty choices")
                return result
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries - 1:
                    break
                text = str(exc).lower()
                delay = API_RETRY_CONFIG["retry_delay"]
                if "400" in text:
                    delay = API_RETRY_CONFIG["retry_delay_400"]
                elif any(code in text for code in ("500", "502", "503", "504")):
                    delay = API_RETRY_CONFIG["retry_delay_500"]
                if API_RETRY_CONFIG.get("exponential_backoff"):
                    delay *= 2 ** attempt
                time.sleep(delay)
        raise ValueError(f"API request failed after {max_retries} attempts: {last_error}")

    def invoke(self, messages: Sequence[Any], **kwargs: Any) -> ModelResponse:
        model_lower = str(self.model).lower()
        # Reasoning chat models (GPT-6 family / o-series) use max_completion_tokens
        # and reject the legacy max_tokens parameter.
        is_reasoning_model = any(
            tag in model_lower for tag in ("gpt-6", "o1", "o3", "o4")
        )
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages),
            "temperature": self.temperature,
            "stream": False,
        }
        if is_reasoning_model:
            payload["max_completion_tokens"] = self.max_tokens
        else:
            payload["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.model_kwargs:
            payload.update(self.model_kwargs)
        payload.update(kwargs)
        result = self._make_api_request(payload)
        usage = _normalize_usage_dict(result)
        metadata: Dict[str, Any] = {}
        if usage:
            metadata["token_usage"] = usage
        if result.get("model"):
            metadata["model_name"] = result["model"]
        return ModelResponse(content=_extract_content(result), response_metadata=metadata, usage_metadata=usage)

def get_vlm(
    provider: str = "openai",
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    top_p: Optional[float] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[int] = None,
    client_max_retries: Optional[int] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> OpenAICompatibleChatModel:
    provider = (provider or "openai").lower()
    openai_compatible_providers = {"openai", "openai_compatible", "vision_http"}
    if provider in openai_compatible_providers:
        key = api_key or os.getenv("OPENAI_API_KEY")
        model = model_name or os.getenv("VLM_MODEL", "gpt-4o")
        return OpenAICompatibleChatModel(
            model=model,
            api_key=key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            timeout=timeout if timeout is not None else _env_int("VLM_API_TIMEOUT", 300),
            client_max_retries=client_max_retries,
            model_kwargs=model_kwargs,
        )
    supported = "', '".join(sorted(openai_compatible_providers))
    raise ValueError(f"Unsupported provider: {provider}. Supported providers: '{supported}'")


create_vlm = get_vlm
