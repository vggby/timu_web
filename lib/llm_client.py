"""LLM 客户端模块"""
from __future__ import annotations

import json
import time
from typing import Callable, Dict, List, Optional

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
        log_callback: Optional[Callable] = None,
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
            "Authorization": f"Bearer {self.api_key[:8]}...",
            "Content-Type": "application/json",
        }
        self.log = log_callback or (lambda level, msg: None)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        for attempt in range(self.max_attempts):
            try:
                self.log('info', f'API 请求 → {self.api_url} | model={self.model} | msg_len={sum(len(m.get("content","")) for m in messages)}')
                t0 = time.time()
                response = requests.post(
                    self.api_url, json=payload, headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    }, timeout=self.timeout
                )
                elapsed = time.time() - t0
                self.log('info', f'API 响应 ← status={response.status_code} | {elapsed:.1f}s | body_len={len(response.text)}')

                if response.status_code != 200:
                    # Try to extract error message
                    try:
                        err_body = response.json()
                        err_msg = err_body.get("error", {}).get("message", "") or str(err_body)
                    except Exception:
                        err_msg = response.text[:300]
                    self.log('error', f'API 错误 {response.status_code}: {err_msg[:300]}')
                    response.raise_for_status()

                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    self.log('error', f'LLM 响应异常: {json.dumps(data, ensure_ascii=False)[:300]}')
                    raise RuntimeError(f"LLM 响应异常：{data}")
                content = choices[0].get("message", {}).get("content")
                if not content:
                    self.log('error', f'LLM 响应消息缺失: {choices[0]}')
                    raise RuntimeError(f"LLM 响应消息缺失：{choices[0]}")

                self.log('info', f'API 成功 | 回复 {len(content)} 字符')
                return content.strip()
            except (requests.RequestException, RuntimeError) as exc:
                if attempt + 1 == self.max_attempts:
                    self.log('error', f'API 最终失败 (重试{attempt+1}次): {exc}')
                    raise
                delay = self.retry_wait * (attempt + 1)
                self.log('warn', f'API 第{attempt+1}次失败，{delay:.1f}s 后重试: {exc}')
                time.sleep(delay)
        raise RuntimeError("LLM 请求失败")
