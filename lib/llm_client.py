"""LLM 客户端模块"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import requests


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        max_retries: int,
        retry_wait: float,
        timeout: float,
        base_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_attempts = max(1, max_retries + 1)
        self.retry_wait = retry_wait
        self.timeout = timeout
        self.base_url = (base_url or "https://api.openai.com").rstrip('/')
        self.api_url = f"{self.base_url}/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        for attempt in range(self.max_attempts):
            try:
                response = requests.post(
                    self.api_url, json=payload, headers=self.headers, timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"LLM 响应异常：{data}")
                content = choices[0].get("message", {}).get("content")
                if not content:
                    raise RuntimeError(f"LLM 响应消息缺失：{choices[0]}")
                return content.strip()
            except (requests.RequestException, RuntimeError) as exc:
                if attempt + 1 == self.max_attempts:
                    raise
                delay = self.retry_wait * (attempt + 1)
                time.sleep(delay)
        raise RuntimeError("LLM 请求失败")
