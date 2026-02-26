"""抓取模块：从芝士架构页面抓取题目数据"""
from __future__ import annotations

import gzip
import json
import re
import zlib
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def build_headers_from_config(config: dict) -> Dict[str, str]:
    """从配置字典构建请求头"""
    base_headers = config.get("headers", {})
    headers: Dict[str, str] = {str(k): str(v) for k, v in base_headers.items() if v is not None}

    defaults = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
    for key, value in defaults.items():
        headers.setdefault(key, value)

    config_cookie = config.get("cookie")
    if config_cookie:
        headers.setdefault("Cookie", str(config_cookie))

    return headers


def decode_body(raw: bytes, encoding_header: str) -> bytes:
    if not encoding_header:
        return raw
    encoding_header = encoding_header.lower()
    if "gzip" in encoding_header:
        return gzip.decompress(raw)
    if "deflate" in encoding_header:
        try:
            return zlib.decompress(raw)
        except zlib.error:
            return zlib.decompress(raw, -zlib.MAX_WBITS)
    if "br" in encoding_header:
        try:
            import brotli
        except ImportError as exc:
            raise SystemExit(
                "检测到 Brotli 压缩（Content-Encoding: br），请安装 brotli 模块。"
            ) from exc
        return brotli.decompress(raw)
    return raw


def fetch_html(url: str, headers: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
    request = Request(url, headers=headers)
    try:
        with urlopen(request) as response:
            raw = response.read()
            content_encoding = response.headers.get("Content-Encoding", "")
            decoded = decode_body(raw, content_encoding)
            encoding = response.headers.get_content_charset() or "utf-8"
            text = decoded.decode(encoding, errors="replace")
            return text, dict(response.headers.items())
    except HTTPError as exc:
        raise SystemExit(f"请求失败：HTTP {exc.code} {exc.reason}") from exc
    except URLError as exc:
        raise SystemExit(f"网络错误：{exc.reason}") from exc


def extract_next_data(html_text: str) -> dict:
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html_text,
        flags=re.DOTALL,
    )
    if not match:
        raise SystemExit("页面中未找到 __NEXT_DATA__ 数据块，无法解析。")
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"解析 JSON 失败：{exc}") from exc


def html_to_text(raw: str) -> str:
    if not raw:
        return ""
    text = unescape(raw)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(raw: str) -> str:
    if not raw:
        return ""
    text = raw.replace("\r", "")
    text = text.replace("\u3000", " ")
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def fetch_and_parse(url: str, config: dict) -> Tuple[str, dict, str]:
    """
    抓取页面并解析，返回 (html_text, next_data, meta_title)。
    config 为配置字典（包含 headers, cookie 等）。
    """
    headers = build_headers_from_config(config)
    html_text, _ = fetch_html(url, headers)
    next_data = extract_next_data(html_text)

    # 提取 meta_title
    page_props = next_data.get("props", {}).get("pageProps", {})
    test_meta = page_props.get("test", {})
    selects = test_meta.get("selects") or []
    cases = test_meta.get("cases") or []
    first_group = selects or cases
    first_paper = first_group[0].get("paper", {}) if first_group else {}
    meta_title = clean_text(
        first_paper.get("name") or test_meta.get("paperName") or "未知试卷"
    )

    return html_text, next_data, meta_title
