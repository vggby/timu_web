"""构建模块：从题目数据生成刷题网站"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from html import escape as html_escape

from lib.fetcher import clean_text, html_to_text
from lib.llm_client import LLMClient


DEFAULT_KNOWLEDGE_PROMPT = (
    "你是一个系统架构师考试辅导老师，针对这个知识点的全部题目，请生成一个高效的学习笔记："
    "请按照以下结构\n"
    "1. 核心概念：使用简洁的语言和图表示意\n"
    "2. 重点难点：常见误区（指出哪些是陷阱）\n"
    "3. 知识要点\n"
    "4. 核心学习技巧和记忆口诀。\n"
    "请使用 Markdown 格式，便于学习和记忆。"
)


def load_prompt(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"提示词文件不存在：{path}")
    return path.read_text(encoding="utf-8").strip()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def question_key(question: dict) -> str:
    return str(question.get("id") or question.get("sort") or question.get("index"))


def normalize_text(value: Optional[str]) -> str:
    return clean_text(value or "")


def build_question_prompt(template: str, meta: dict, question: dict) -> str:
    lines: List[str] = []
    lines.append(f"试卷：{meta.get('paper_name', '未知试卷')}")
    lines.append(f"知识点：{question['knowledge_point']}")
    lines.append(f"题号：{question.get('sort') or question['index']}")
    lines.append(f"题干：{question['prompt']}")
    for sub in question["sub_questions"]:
        lines.append(f"子题{sub['label']}")
        if sub.get("question"):
            lines.append(f"子题题目：{sub['question']}")
        for opt in sub["options"]:
            lines.append(f"{opt['label']}. {opt['text']}")
        if sub["correct_letters"]:
            lines.append(f"参考答案：{''.join(sub['correct_letters'])}")
        if sub["correct_texts"]:
            lines.append(f"答案对应内容：{' / '.join(sub['correct_texts'])}")
        if sub.get("correct_answer"):
            lines.append(f"参考答案：{sub['correct_answer']}")
        if sub["official_analysis"]:
            lines.append(f"官方解析：{sub['official_analysis']}")
    if question["tags"]:
        lines.append(f"标签：{''.join(question['tags'])}")
    return template + "\n\n" + "\n".join(lines)


def build_knowledge_prompt(template: str, knowledge_point: str, meta: dict, questions: List[dict], question_responses: Dict[str, dict]) -> str:
    lines: List[str] = []
    lines.append(f"知识点：{knowledge_point}")
    lines.append(f"试卷：{meta.get('paper_name', '未知试卷')}")
    lines.append("以下是该知识点对应的全部题目：")
    for q in questions:
        key = question_key(q)
        lines.append("\n---\n")
        lines.append(f"题号：{q.get('sort') or q['index']}")
        lines.append(f"题干：{q['prompt']}")
        for sub in q["sub_questions"]:
            lines.append(f"子题{sub['label']}")
            if sub.get("question"):
                lines.append(f"子题题目：{sub['question']}")
            for opt in sub["options"]:
                lines.append(f"{opt['label']}. {opt['text']}")
            if sub["correct_letters"]:
                lines.append(f"参考答案：{''.join(sub['correct_letters'])}")
            if sub.get("correct_answer"):
                lines.append(f"参考答案：{sub['correct_answer']}")
            if sub["official_analysis"]:
                lines.append(f"官方解析：{sub['official_analysis']}")
        response_block = question_responses.get(key, "")
        if isinstance(response_block, dict):
            resp_text = response_block.get("response", "")
        else:
            resp_text = str(response_block).strip()
        if resp_text:
            lines.append("AI 学习笔记：")
            lines.append(resp_text)
    return template + "\n\n" + "\n".join(lines)


def extract_questions(next_data: dict) -> Tuple[dict, List[dict], Dict[str, List[dict]]]:
    page_props = next_data.get("props", {}).get("pageProps", {})
    test_meta = page_props.get("test", {})
    selects = test_meta.get("selects") or []
    cases = test_meta.get("cases") or []

    if not selects and not cases:
        raise SystemExit("未找到页面中的题目列表")

    source_list = selects if selects else cases

    first_paper = (source_list[0] or {}).get("paper", {}) if source_list else {}
    meta = {
        "paper_name": normalize_text(first_paper.get("name") or test_meta.get("paperName") or "未知试卷"),
        "subject": test_meta.get("subject"),
        "type": test_meta.get("type"),
        "item_count": len(source_list),
    }

    questions: List[dict] = []
    knowledge_map: Dict[str, List[dict]] = {}

    for idx, select in enumerate(source_list, 1):
        kp = normalize_text(select.get("kpName") or "未分类")
        prompt = normalize_text(select.get("promptMd")) or html_to_text(select.get("prompt") or "")
        tags = [normalize_text(tag.get("name")) for tag in select.get("tagList") or [] if tag.get("name")]
        sub_md_list = select.get("subQuestionsMd") or []
        sub_html_list = select.get("subQuestions") or []
        sub_questions: List[dict] = []

        for sub_idx, md_entry in enumerate(sub_md_list, 1):
            html_entry = sub_html_list[sub_idx - 1] if sub_idx - 1 < len(sub_html_list) else {}
            options_md = md_entry.get("options") or []
            options_html = html_entry.get("options") or []
            options: List[dict] = []
            correct_letters: List[str] = []
            correct_texts: List[str] = []
            option_source = options_md or options_html
            for opt_idx, opt in enumerate(option_source):
                text = opt.get("text")
                if text is None and opt_idx < len(options_html):
                    text = options_html[opt_idx].get("text")
                text_normalized = normalize_text(text)
                label = chr(ord("A") + opt_idx)
                is_correct = bool(opt.get("isCorrect"))
                options.append({
                    "label": label,
                    "text": text_normalized,
                    "is_correct": is_correct,
                })
                if is_correct:
                    correct_letters.append(label)
                    correct_texts.append(text_normalized)
            analysis = md_entry.get("analysis") or ""
            if not analysis and html_entry:
                analysis = html_to_text(html_entry.get("analysis") or "")
            sub_question_text = md_entry.get("question") or ""
            if not sub_question_text and html_entry:
                sub_question_text = html_to_text(html_entry.get("question") or "")
            correct_answer = md_entry.get("correctAnswer") or ""
            if not correct_answer and html_entry:
                correct_answer = html_to_text(html_entry.get("correctAnswer") or "")
            sub_questions.append({
                "label": str(sub_idx),
                "question": sub_question_text.strip(),
                "correct_answer": correct_answer.strip(),
                "options": options,
                "correct_letters": correct_letters,
                "correct_texts": correct_texts,
                "official_analysis": analysis.strip(),
            })

        question_entry = {
            "id": select.get("id"),
            "index": idx,
            "sort": select.get("sort"),
            "knowledge_point": kp,
            "prompt": prompt,
            "difficulty": select.get("difficultyLevel"),
            "tags": tags,
            "sub_questions": sub_questions,
        }
        questions.append(question_entry)
        knowledge_map.setdefault(kp, []).append(question_entry)

    return meta, questions, knowledge_map


def generate_html(site_data: dict, output_path: Path) -> None:
    data_json = json.dumps(site_data, ensure_ascii=False)
    data_json = data_json.replace('</', '<\/')

    def format_inline(text: str) -> str:
        # First extract images and protect them before escaping
        img_placeholders = []
        def replace_img(m):
            idx = len(img_placeholders)
            img_placeholders.append(m.group(0))
            return f'__IMG_PLACEHOLDER_{idx}__'

        # Protect markdown images: ![alt](url)
        text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_img, text)
        # Protect bare image URLs after !
        text = re.sub(r'!(https?://[^\s<>"\'\[\]]+(?:\.[a-zA-Z]{2,}|/[^\s<>"\'\[\]]*)?)', replace_img, text)

        escaped = html_escape(text)

        # Now restore images as proper HTML
        for idx, orig in enumerate(img_placeholders):
            placeholder = f'__IMG_PLACEHOLDER_{idx}__'
            # Check if it's ![alt](url) format
            m = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', orig)
            if m:
                alt, url = html_escape(m.group(1)), html_escape(m.group(2))
                img_html = f'<img src="{url}" alt="{alt}" style="max-width: 100%; height: auto;" />'
            else:
                # bare !url format
                m2 = re.match(r'!(https?://\S+)', orig)
                url = html_escape(m2.group(1)) if m2 else ''
                img_html = f'<img src="{url}" alt="图片" style="max-width: 100%; height: auto;" />' if url else ''
            escaped = escaped.replace(placeholder, img_html)

        escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)
        escaped = re.sub(r'__(.+?)__', r'<strong>\1</strong>', escaped)
        escaped = re.sub(r'`([^`]+)`', r'<code>\1</code>', escaped)
        escaped = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', escaped)
        return escaped

    def render_markdown_basic(raw: str) -> str:
        if not raw:
            return '<p>（尚未生成，稍后重试）</p>'
        normalized = raw.replace('\r\n', '\n').replace('\r', '\n')
        lines = normalized.split('\n')
        html_parts: List[str] = []
        in_ul = False
        in_ol = False

        def close_lists() -> None:
            nonlocal in_ul, in_ol
            if in_ul:
                html_parts.append('</ul>')
                in_ul = False
            if in_ol:
                html_parts.append('</ol>')
                in_ol = False

        def render_table(table_lines: List[str]) -> str:
            if not table_lines:
                return ''

            def split_row(row: str) -> List[str]:
                return [format_inline(cell.strip()) for cell in row.strip('|').split('|')]

            header_cells = split_row(table_lines[0])
            alignments = ['left'] * len(header_cells)
            body_lines = table_lines[1:]
            if body_lines:
                first_body = [cell.strip() for cell in body_lines[0].strip('|').split('|')]
                if len(first_body) == len(header_cells) and all(re.fullmatch(r':?[-]{3,}:?', cell) for cell in first_body):
                    alignments = []
                    for cell in first_body:
                        left = cell.startswith(':')
                        right = cell.endswith(':')
                        if left and right:
                            alignments.append('center')
                        elif right:
                            alignments.append('right')
                        else:
                            alignments.append('left')
                    body_lines = body_lines[1:]
            header_html = '<tr>' + ''.join(
                f"<th style='text-align: {alignments[idx if idx < len(alignments) else 0]}'>{cell}</th>"
                for idx, cell in enumerate(header_cells)
            ) + '</tr>'
            body_html_parts: List[str] = []
            for row in body_lines:
                cells = split_row(row)
                row_html = '<tr>'
                for idx, cell in enumerate(cells):
                    align = alignments[idx] if idx < len(alignments) else 'left'
                    row_html += f"<td style='text-align: {align}'>{cell}</td>"
                row_html += '</tr>'
                body_html_parts.append(row_html)
            body_html = ''.join(body_html_parts) or '<tr>' + ''.join(
                f"<td style='text-align: {alignments[idx if idx < len(alignments) else 0]}'>&nbsp;</td>"
                for idx in range(len(header_cells))
            ) + '</tr>'
            return f"<table><thead>{header_html}</thead><tbody>{body_html}</tbody></table>"

        idx = 0
        total = len(lines)
        while idx < total:
            stripped = lines[idx].strip()
            if not stripped:
                close_lists()
                idx += 1
                continue
            if stripped.startswith('|') and stripped.endswith('|') and '|' in stripped.strip('|'):
                close_lists()
                table_block: List[str] = []
                while idx < total:
                    current = lines[idx].strip()
                    if not (current.startswith('|') and current.endswith('|') and '|' in current.strip('|')):
                        break
                    table_block.append(current)
                    idx += 1
                if table_block:
                    html_parts.append(render_table(table_block))
                continue
            if re.match(r'^\d+\.\s+', stripped):
                if not in_ol:
                    close_lists()
                    html_parts.append('<ol>')
                    in_ol = True
                content = re.sub(r'^\d+\.\s+', '', stripped, count=1)
                html_parts.append(f'<li>{format_inline(content)}</li>')
                idx += 1
                continue
            if re.match(r'^[-*+]\s+', stripped):
                if not in_ul:
                    close_lists()
                    html_parts.append('<ul>')
                    in_ul = True
                content = stripped[1:].strip()
                html_parts.append(f'<li>{format_inline(content)}</li>')
                idx += 1
                continue
            heading_match = re.match(r'^(#{1,4})\s+(.+)$', stripped)
            if heading_match:
                close_lists()
                level = min(len(heading_match.group(1)) + 1, 5)
                html_parts.append(f'<h{level} class="md-heading">{format_inline(heading_match.group(2))}</h{level}>')
                idx += 1
                continue
            if stripped in ('---', '***', '___'):
                close_lists()
                html_parts.append('<hr class="md-hr" />')
                idx += 1
                continue
            close_lists()
            html_parts.append(f'<p>{format_inline(stripped)}</p>')
            idx += 1

        close_lists()
        return ''.join(html_parts)

    knowledge_cards = []
    for kp in site_data.get('knowledge_points', []):
        name = html_escape(kp.get('name') or '未分类')
        count = len(kp.get('related_questions') or [])
        summary_markdown = kp.get('summary_markdown') or ''
        if isinstance(summary_markdown, dict):
            summary_markdown = summary_markdown.get('response', '')
        if summary_markdown and summary_markdown.strip():
            summary_html = render_markdown_basic(summary_markdown)
            summary_block = (
                f"<details class='knowledge-details'><summary>查看大模型总结</summary><div class='markdown'>{summary_html}</div></details>"
            )
        else:
            summary_block = "<div class='markdown empty'><p>（尚未生成，稍后重试）</p></div>"
        knowledge_cards.append(
            f"""
            <article class='card knowledge-card' style='cursor:pointer' onclick="jumpToKnowledge('{name}')">
              <div class='knowledge-header'>
                <h3>{name}</h3>
                <span class='badge badge-accent kp-count-btn' title='点击刷此知识点题库' onclick="event.stopPropagation();jumpToKnowledge('{name}')">共 {count} 题 ▶</span>
              </div>
              {summary_block}
            </article>
            """.strip()
        )

    question_cards = []
    for q in site_data.get('questions', []):
        knowledge_point = html_escape(q.get('knowledge_point') or '未分类')
        prompt = format_inline(q.get('prompt') or '')
        difficulty = q.get('difficulty')
        badge_difficulty = (
            f"<span class='badge'>难度：{difficulty}</span>" if difficulty not in (None, '') else ''
        )
        header = (
            f"""
            <div class='question-header'>
              <h3>第{q.get('index')}题（题号：{q.get('sort') or q.get('index')}）</h3>
              <span class='badge'>{knowledge_point}</span>
              {badge_difficulty}
              <button class='ai-chat-btn' onclick="openAiChat({q.get('index')})">🤖 AI对话</button>
            </div>
            <p>{prompt}</p>
            """.strip()
        )
        sub_sections = []
        for sub_idx, sub in enumerate(q.get('sub_questions', [])):
            options = []
            correct_count = len([opt for opt in sub.get('options', []) if opt.get('is_correct')])
            is_multiple = correct_count > 1
            input_type = 'checkbox' if is_multiple else 'radio'
            question_id = f"q{q.get('id', q.get('index'))}_{sub_idx}"
            sub_question_text = sub.get('question') or ''
            sub_question_html = f"<div class='sub-question-text'>{render_markdown_basic(sub_question_text)}</div>" if sub_question_text else ''
            correct_answer_text = sub.get('correct_answer') or ''
            correct_answer_html = (
                f"<details class='rich-details answer-analysis' style='display: none;'><summary>参考答案</summary><div class='markdown'>{render_markdown_basic(correct_answer_text)}</div></details>"
                if correct_answer_text else ''
            )
            for opt in sub.get('options', []):
                option_text = html_escape(opt.get('text') or '')
                option_id = f"{question_id}_{opt.get('label')}"
                options.append(f"""
                    <li class="option-item">
                        <label for="{option_id}" class="option-label">
                            <input type="{input_type}" id="{option_id}" name="{question_id}" value="{opt.get('label')}" class="option-input" />
                            <span class="option-text">{opt.get('label')}. {option_text}</span>
                        </label>
                    </li>
                """)
            options_html = '\n'.join(options)
            answers = sub.get('correct_letters') or []
            answer_html = (
                f"<p class='correct-answer' style='display: none;'><strong>参考答案：</strong>{'、'.join(answers)}</p>" if answers else ''
            )
            analysis_html = render_markdown_basic(sub.get('official_analysis') or '')
            has_options = len(sub.get('options', [])) > 0
            if has_options:
                interaction_html = f"""
                  <ul class='options interactive-options'>{options_html}</ul>
                  <div class="question-actions">
                    <button class="submit-btn" onclick="submitAnswer('{question_id}')">提交答案</button>
                    <button class="show-answer-btn" onclick="showAnswer('{question_id}')" style="display: none;">查看答案</button>
                  </div>
                """
            else:
                interaction_html = f"""
                  <div class="question-actions">
                    <button class="show-answer-btn" onclick="showAnswer('{question_id}')" style="display: inline-block;">查看答案</button>
                  </div>
                """
            sub_sections.append(
                f"""
                <div class='sub-question' data-question-id="{question_id}" data-correct-answers="{','.join(answers)}" data-is-multiple="{str(is_multiple).lower()}">
                  <h4>子题{sub.get('label')}</h4>
                  {sub_question_html}
                  {interaction_html}
                  <div class="answer-feedback" style="display: none;"></div>
                  {answer_html}
                  {correct_answer_html}
                  <details class='rich-details answer-analysis' style='display: none;'><summary>官方解析</summary><div class='markdown'>{analysis_html}</div></details>
                </div>
                """.strip()
            )
        sub_html = '\n'.join(sub_sections)
        model_response = q.get('model_response') or ''
        if isinstance(model_response, dict):
            model_response = model_response.get('response', '')
        if model_response:
            ai_note = (
                f"""
                <details class='rich-details'><summary>AI 记忆笔记</summary><div class='markdown'>{render_markdown_basic(model_response)}</div></details>
                """.strip()
            )
        else:
            ai_note = "<details class='rich-details'><summary>AI 记忆笔记</summary><div class='markdown empty'>暂无内容</div></details>"
        question_cards.append(
            f"""
            <article class='card question-card'>
              {header}
              {sub_html}
              {ai_note}
            </article>
            """.strip()
        )

    knowledge_initial = '\n'.join(knowledge_cards)
    questions_initial = '\n'.join(question_cards)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{site_data['meta'].get('paper_name', '刷题笔记')}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <link rel="preconnect" href="https://cdn.jsdelivr.net" />
  <script src="https://cdn.jsdelivr.net/npm/markdown-it@13/dist/markdown-it.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.6.1/mermaid.min.js"></script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" />
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
  <script>
    function renderAllMath() {{
      if (typeof renderMathInElement === 'function') {{
        renderMathInElement(document.body, {{
          delimiters: [
            {{left: '$$', right: '$$', display: true}},
            {{left: '$', right: '$', display: false}},
            {{left: '\\\\(', right: '\\\\)', display: false}},
            {{left: '\\\\[', right: '\\\\]', display: true}}
          ],
          throwOnError: false
        }});
      }}
    }}
  </script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    :root {{
      --font-sans: 'Inter', 'Noto Sans SC', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      --bg: #0a0a0f;
      --bg-subtle: #111118;
      --card: #16161f;
      --card-hover: #1c1c28;
      --accent: #7c6ff7;
      --accent-light: #a89cf8;
      --accent-bg: rgba(124, 111, 247, 0.12);
      --accent-bg-hover: rgba(124, 111, 247, 0.2);
      --green: #34d399;
      --green-bg: rgba(52, 211, 153, 0.12);
      --green-border: rgba(52, 211, 153, 0.35);
      --red: #f87171;
      --red-bg: rgba(248, 113, 113, 0.12);
      --red-border: rgba(248, 113, 113, 0.35);
      --orange: #fbbf24;
      --text: #e2e8f0;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --border: rgba(255,255,255,0.08);
      --border-hover: rgba(255,255,255,0.15);
      --shadow-sm: 0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2);
      --shadow-md: 0 4px 12px rgba(0,0,0,0.4), 0 2px 4px rgba(0,0,0,0.3);
      --shadow-lg: 0 10px 30px rgba(0,0,0,0.5), 0 4px 8px rgba(0,0,0,0.3);
      --radius: 16px;
      --radius-sm: 10px;
      --radius-xs: 6px;
      --glass: rgba(255,255,255,0.04);
      --glass-border: rgba(255,255,255,0.08);
    }}
    [data-theme="light"] {{
      --bg: #f0f4f8;
      --bg-subtle: #e8edf2;
      --card: #ffffff;
      --card-hover: #f8f9fc;
      --accent: #6c5ce7;
      --accent-light: #a29bfe;
      --accent-bg: rgba(108, 92, 231, 0.10);
      --accent-bg-hover: rgba(108, 92, 231, 0.18);
      --green: #10b981;
      --green-bg: rgba(16, 185, 129, 0.10);
      --green-border: rgba(16, 185, 129, 0.30);
      --red: #ef4444;
      --red-bg: rgba(239, 68, 68, 0.10);
      --red-border: rgba(239, 68, 68, 0.30);
      --orange: #f59e0b;
      --text: #1e293b;
      --text-secondary: #475569;
      --text-muted: #94a3b8;
      --border: rgba(0,0,0,0.08);
      --border-hover: rgba(0,0,0,0.15);
      --shadow-sm: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.05);
      --shadow-md: 0 4px 12px rgba(0,0,0,0.10), 0 2px 4px rgba(0,0,0,0.06);
      --shadow-lg: 0 10px 30px rgba(0,0,0,0.12), 0 4px 8px rgba(0,0,0,0.08);
      --glass: rgba(0,0,0,0.02);
      --glass-border: rgba(0,0,0,0.08);
    }}
    /* ===== Theme toggle button ===== */
    .theme-toggle {{
      position: absolute;
      top: 16px;
      right: 16px;
      z-index: 10;
      background: rgba(255,255,255,0.15);
      border: 1.5px solid rgba(255,255,255,0.25);
      border-radius: 50px;
      padding: 6px 14px;
      cursor: pointer;
      font-size: 0.85rem;
      font-weight: 600;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 6px;
      backdrop-filter: blur(8px);
      transition: background 0.2s, transform 0.15s;
      font-family: var(--font-sans);
    }}
    .theme-toggle:hover {{
      background: rgba(255,255,255,0.25);
      transform: scale(1.04);
    }}
    [data-theme="light"] .theme-toggle {{
      background: rgba(0,0,0,0.12);
      border-color: rgba(0,0,0,0.18);
      color: #fff;
    }}
    [data-theme="light"] .theme-toggle:hover {{
      background: rgba(0,0,0,0.20);
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-sans);
      line-height: 1.6;
      -webkit-font-smoothing: antialiased;
    }}
    main {{
      max-width: 960px;
      margin: 0 auto;
      padding: 24px 20px 60px;
    }}

    /* ===== Header ===== */
    .page-header {{
      background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #a855f7 100%);
      border-radius: var(--radius);
      padding: 32px 28px;
      margin-bottom: 24px;
      color: #fff;
      box-shadow: var(--shadow-lg);
      position: relative;
      overflow: hidden;
    }}
    .page-header::before {{
      content: '';
      position: absolute;
      top: -50%;
      right: -20%;
      width: 300px;
      height: 300px;
      background: rgba(255,255,255,0.08);
      border-radius: 50%;
    }}
    .page-header::after {{
      content: '';
      position: absolute;
      bottom: -30%;
      left: -10%;
      width: 200px;
      height: 200px;
      background: rgba(255,255,255,0.05);
      border-radius: 50%;
    }}
    .page-header h1 {{
      margin: 0 0 8px;
      font-size: 1.75rem;
      font-weight: 700;
      position: relative;
      z-index: 1;
    }}
    .page-header .meta-line {{
      margin: 0;
      font-size: 0.9rem;
      opacity: 0.85;
      position: relative;
      z-index: 1;
    }}

    /* ===== Controls ===== */
    .controls-bar {{
      display: flex;
      gap: 12px;
      margin-bottom: 20px;
      flex-wrap: wrap;
    }}
    .controls-bar select,
    .controls-bar input {{
      flex: 1 1 200px;
      min-width: 0;
      padding: 12px 16px;
      border-radius: var(--radius-sm);
      border: 1.5px solid var(--border);
      background: var(--card);
      font-size: 0.95rem;
      font-family: var(--font-sans);
      color: var(--text);
      transition: border-color 0.2s, box-shadow 0.2s;
      box-shadow: var(--shadow-sm);
      -webkit-appearance: none;
    }}
    .controls-bar select:focus,
    .controls-bar input:focus {{
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
    }}

    /* ===== Stats Ring ===== */
    .stats-card {{
      background: var(--card);
      border-radius: var(--radius);
      padding: 24px;
      margin-bottom: 20px;
      box-shadow: var(--shadow-md);
    }}
    .stats-card h2 {{
      margin: 0 0 20px;
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .stats-layout {{
      display: flex;
      align-items: center;
      gap: 32px;
      flex-wrap: wrap;
    }}
    .ring-container {{
      flex-shrink: 0;
      position: relative;
      width: 120px;
      height: 120px;
    }}
    .ring-container svg {{
      transform: rotate(-90deg);
    }}
    .ring-container .ring-label {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      text-align: center;
    }}
    .ring-container .ring-pct {{
      font-size: 1.5rem;
      font-weight: 700;
      color: var(--accent);
      display: block;
      line-height: 1.2;
    }}
    .ring-container .ring-text {{
      font-size: 0.7rem;
      color: var(--text-secondary);
    }}
    .stats-numbers {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
      flex: 1;
      min-width: 200px;
    }}
    .sn-item {{
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 14px 16px;
      text-align: center;
    }}
    .sn-val {{
      font-size: 1.4rem;
      font-weight: 700;
      line-height: 1.2;
    }}
    .sn-val.green {{ color: var(--green); }}
    .sn-val.red {{ color: var(--red); }}
    .sn-val.accent {{ color: var(--accent); }}
    .sn-val.orange {{ color: var(--orange); }}
    .sn-lbl {{
      font-size: 0.78rem;
      color: var(--text-secondary);
      margin-top: 2px;
    }}
    .stats-progress {{
      margin-top: 16px;
    }}
    .stats-progress-bar {{
      width: 100%;
      height: 6px;
      background: var(--border);
      border-radius: 3px;
      overflow: hidden;
    }}
    .stats-progress-fill {{
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--accent-light));
      border-radius: 3px;
      transition: width 0.4s ease;
    }}
    .stats-progress-text {{
      font-size: 0.78rem;
      color: var(--text-secondary);
      margin-top: 6px;
      text-align: right;
    }}

    /* ===== Knowledge Section ===== */
    .section-title {{
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--text);
      margin: 0 0 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .knowledge-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}
    .knowledge-card {{
      background: var(--card);
      border-radius: var(--radius);
      padding: 20px;
      box-shadow: var(--shadow-sm);
      border-left: 4px solid var(--accent-light);
      transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
      cursor: default;
    }}
    .knowledge-card:nth-child(3n+1) {{ border-left-color: #4f46e5; }}
    .knowledge-card:nth-child(3n+2) {{ border-left-color: #0ea5e9; }}
    .knowledge-card:nth-child(3n+3) {{ border-left-color: #8b5cf6; }}
    .knowledge-card:hover {{
      transform: translateY(-2px);
      box-shadow: var(--shadow-md);
    }}
    .knowledge-card.active {{
      border-left-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(79,70,229,0.2), var(--shadow-md);
    }}
    .knowledge-card.expanded {{
      grid-column: 1 / -1;
      transition: grid-column 0s;
    }}
    .knowledge-card.expanded .knowledge-details {{
      max-height: none;
    }}
    .knowledge-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .knowledge-header h3 {{
      margin: 0;
      font-size: 1rem;
      font-weight: 600;
      color: var(--text);
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 600;
      white-space: nowrap;
    }}
    .badge-accent {{
      background: var(--accent-bg);
      color: var(--accent);
    }}
    .kp-count-btn {{
      cursor: pointer;
      transition: background 0.18s, transform 0.15s;
      user-select: none;
    }}
    .kp-count-btn:hover {{
      background: var(--accent);
      color: #fff;
      transform: scale(1.06);
    }}
    .badge-green {{
      background: var(--green-bg);
      color: var(--green);
    }}
    .knowledge-details {{
      margin-top: 12px;
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 10px 14px;
    }}
    .knowledge-details summary {{
      cursor: pointer;
      font-weight: 600;
      font-size: 0.88rem;
      color: var(--accent);
      user-select: none;
      list-style: none;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .knowledge-details summary::before {{
      content: '▶';
      font-size: 0.7rem;
      transition: transform 0.2s;
      display: inline-block;
    }}
    .knowledge-details[open] summary::before {{
      transform: rotate(90deg);
    }}
    .knowledge-details .markdown {{
      margin-top: 10px;
    }}
    .knowledge-details[open] summary {{
      margin-bottom: 8px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
    }}

    /* ===== Question Cards ===== */
    .question-card {{
      background: var(--card);
      border-radius: var(--radius);
      padding: 24px;
      margin-bottom: 16px;
      box-shadow: var(--shadow-sm);
      transition: box-shadow 0.2s;
    }}
    .question-card:hover {{
      box-shadow: var(--shadow-md);
    }}
    .question-header {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-bottom: 12px;
    }}
    .question-header h3 {{
      margin: 0;
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--text);
    }}
    .question-prompt {{
      margin: 0 0 16px;
      color: var(--text);
      line-height: 1.7;
    }}
    .sub-question {{
      margin-top: 12px;
      padding-top: 16px;
      border-top: 1px solid var(--border);
    }}
    .sub-question h4 {{
      margin: 0 0 8px;
      font-size: 0.92rem;
      font-weight: 600;
      color: var(--text);
    }}
    .sub-question-text {{
      margin-bottom: 12px;
      color: var(--text-secondary);
    }}

    /* ===== Options ===== */
    ul.options {{
      list-style: none;
      padding: 0;
      margin: 0 0 12px;
    }}
    .option-item {{
      margin-bottom: 8px;
      border-radius: var(--radius-sm);
      border: 1.5px solid var(--border);
      background: rgba(255,255,255,0.03);
      transition: all 0.2s ease;
      overflow: hidden;
    }}
    .option-item:hover {{
      border-color: var(--accent-light);
      background: var(--accent-bg);
      transform: translateX(2px);
    }}
    .option-label {{
      display: flex;
      align-items: center;
      padding: 12px 16px;
      cursor: pointer;
      gap: 12px;
    }}
    .option-input {{
      flex-shrink: 0;
      width: 18px;
      height: 18px;
      accent-color: var(--accent);
    }}
    .option-text {{
      flex: 1;
      color: var(--text);
      line-height: 1.5;
      font-size: 0.95rem;
    }}
    .option-item.selected {{
      border-color: var(--accent);
      background: var(--accent-bg);
    }}
    .option-item.correct {{
      border-color: var(--green);
      background: var(--green-bg);
      animation: correctPulse 0.5s ease;
    }}
    .option-item.incorrect {{
      border-color: var(--red);
      background: var(--red-bg);
      animation: shake 0.4s ease;
    }}
    @keyframes shake {{
      0%, 100% {{ transform: translateX(0); }}
      20% {{ transform: translateX(-6px); }}
      40% {{ transform: translateX(6px); }}
      60% {{ transform: translateX(-4px); }}
      80% {{ transform: translateX(4px); }}
    }}
    @keyframes correctPulse {{
      0% {{ transform: scale(1); }}
      50% {{ transform: scale(1.01); }}
      100% {{ transform: scale(1); }}
    }}
    @keyframes bounceIn {{
      0% {{ opacity: 0; transform: scale(0.9); }}
      60% {{ transform: scale(1.02); }}
      100% {{ opacity: 1; transform: scale(1); }}
    }}
    @keyframes toastIn {{
      from {{ opacity: 0; transform: translateX(-50%) translateY(12px); }}
      to   {{ opacity: 1; transform: translateX(-50%) translateY(0); }}
    }}
    @keyframes fadeSlideUp {{
      from {{ opacity: 0; transform: translateY(8px); }}
      to   {{ opacity: 1; transform: translateY(0); }}
    }}

    /* ===== Toast ===== */
    #_toast {{
      position: fixed;
      bottom: 32px;
      left: 50%;
      transform: translateX(-50%);
      padding: 12px 24px;
      border-radius: 40px;
      font-size: 0.9rem;
      font-weight: 500;
      pointer-events: none;
      opacity: 0;
      z-index: 9999;
      white-space: nowrap;
      transition: opacity 0.3s;
    }}
    #_toast._toast-show {{
      animation: toastIn 0.25s ease forwards;
    }}
    #_toast._toast-info  {{ background: var(--accent);  color: #fff; }}
    #_toast._toast-warn  {{ background: var(--orange);  color: #000; }}
    #_toast._toast-ok    {{ background: var(--green);   color: #000; }}
    #_toast._toast-err   {{ background: var(--red);     color: #fff; }}

    /* ===== Answer Feedback ===== */
    .answer-feedback {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 12px 16px;
      border-radius: var(--radius-sm);
      font-weight: 600;
      font-size: 0.95rem;
      animation: fadeSlideUp 0.3s ease;
    }}
    .feedback-icon {{ font-size: 1.1rem; }}
    .feedback-msg {{ flex: 1; }}

    /* ===== Buttons ===== */
    .question-actions {{
      margin: 16px 0;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .btn {{
      padding: 10px 22px;
      border: none;
      border-radius: var(--radius-sm);
      font-size: 0.9rem;
      font-weight: 600;
      font-family: var(--font-sans);
      cursor: pointer;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .btn-primary {{
      background: var(--accent);
      color: #fff;
      box-shadow: 0 2px 6px rgba(79,70,229,0.3);
    }}
    .btn-primary:hover {{
      background: #6d62e0;
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(79,70,229,0.35);
    }}
    .btn-primary:disabled {{
      background: var(--text-muted);
      cursor: not-allowed;
      transform: none;
      box-shadow: none;
    }}
    .btn-warning {{
      background: var(--orange);
      color: #fff;
      box-shadow: 0 2px 6px rgba(245,158,11,0.3);
    }}
    .btn-warning:hover {{
      background: #e4a320;
      transform: translateY(-1px);
    }}
    .btn-success {{
      background: var(--green);
      color: #fff;
      box-shadow: 0 2px 6px rgba(16,185,129,0.3);
    }}
    .btn-success:hover {{
      background: #2ab77f;
      transform: translateY(-1px);
    }}
    .btn-outline {{
      background: var(--card);
      color: var(--accent);
      border: 1.5px solid var(--accent);
    }}
    .btn-outline:hover {{
      background: var(--accent-bg);
    }}
    .btn-outline:disabled {{
      color: var(--text-muted);
      border-color: var(--border);
      background: transparent;
      cursor: not-allowed;
      opacity: 0.4;
    }}
    .btn-ghost {{
      background: transparent;
      border: 1.5px solid transparent;
      color: var(--text-secondary);
      font-size: 0.85rem;
    }}
    .btn-ghost:hover {{
      color: var(--text);
      background: var(--glass);
      border-color: var(--border);
    }}

    /* ===== Feedback ===== */
    .answer-feedback.correct {{
      background: var(--green-bg);
      color: var(--green);
      border: 1px solid var(--green-border);
    }}
    .answer-feedback.incorrect {{
      background: var(--red-bg);
      color: var(--red);
      border: 1px solid var(--red-border);
    }}
    .correct-answer {{
      background: var(--green-bg);
      padding: 10px 14px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--green-border);
      margin: 8px 0;
      font-size: 0.95rem;
    }}

    /* ===== Quiz Navigation ===== */
    .quiz-progress-bar {{
      background: var(--card);
      border-radius: var(--radius);
      padding: 20px 24px;
      margin-bottom: 16px;
      box-shadow: var(--shadow-sm);
    }}
    .qp-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }}
    .qp-label {{
      font-weight: 600;
      font-size: 0.95rem;
      color: var(--text);
    }}
    .qp-count {{
      font-size: 0.85rem;
      color: var(--text-secondary);
    }}
    .qp-bar {{
      width: 100%;
      height: 8px;
      background: var(--border);
      border-radius: 4px;
      overflow: hidden;
    }}
    .qp-fill {{
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--accent-light));
      border-radius: 4px;
      transition: width 0.4s ease;
    }}
    .quiz-nav {{
      display: flex;
      justify-content: center;
      gap: 10px;
      margin-top: 20px;
      padding-top: 20px;
      border-top: 1px solid var(--border);
      flex-wrap: wrap;
    }}
    .quiz-mode-toggle {{
      text-align: center;
      margin-bottom: 16px;
    }}

    /* ===== Details / Panels ===== */
    .rich-details {{
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 10px 14px;
      margin-top: 10px;
    }}
    .rich-details summary {{
      cursor: pointer;
      font-weight: 600;
      font-size: 0.88rem;
      color: var(--accent-light);
      user-select: none;
      transition: color 0.2s;
      list-style: none;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .rich-details summary::before {{
      content: '▶';
      font-size: 0.7rem;
      transition: transform 0.2s;
      display: inline-block;
    }}
    .rich-details[open] summary::before {{
      transform: rotate(90deg);
    }}
    .rich-details summary:hover {{
      color: var(--accent);
    }}
    .rich-details[open] summary {{
      margin-bottom: 8px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border);
    }}
    .rich-details .markdown {{
      margin-top: 8px;
    }}
    details summary::-webkit-details-marker {{ display: none; }}
    .answer-analysis {{
      margin-top: 10px;
    }}
    /* ===== Markdown ===== */
    .markdown {{
      line-height: 1.7;
      color: inherit;
      font-size: 0.95rem;
    }}
    .markdown h1, .markdown h2, .markdown h3, .markdown h4 {{
      margin: 1em 0 0.5em;
      font-weight: 700;
    }}
    .markdown h1, .md-heading.h2-equiv {{ font-size: 1.3rem; color: var(--text); }}
    .markdown h2, .md-heading {{ font-size: 1.15rem; color: var(--text); }}
    .markdown h3 {{ font-size: 1.05rem; }}
    .md-heading {{ margin: 1em 0 0.4em; font-weight: 700; }}
    .md-hr {{ border: none; border-top: 1px solid var(--border); margin: 1em 0; }}
    .markdown p {{ margin: 0.5em 0; }}
    .markdown ul, .markdown ol {{ padding-left: 1.5em; margin: 0.5em 0; }}
    .markdown li {{ margin-bottom: 0.3em; }}
    .markdown table {{
      width: 100%;
      border-collapse: collapse;
      margin: 0.75rem 0;
      font-size: 0.9rem;
    }}
    .markdown th, .markdown td {{
      border: 1px solid var(--border);
      padding: 8px 12px;
    }}
    .markdown th {{
      background: var(--accent-bg);
      font-weight: 600;
    }}
    .markdown code {{
      background: rgba(99,102,241,0.1);
      border-radius: 4px;
      padding: 2px 6px;
      font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
      font-size: 0.88em;
    }}
    .markdown pre {{
      background: #0d0d14;
      color: #e2e8f0;
      border: 1px solid var(--border);
      padding: 16px;
      border-radius: var(--radius-sm);
      overflow-x: auto;
      font-size: 0.88rem;
    }}
    .markdown pre code {{
      background: none;
      padding: 0;
      color: inherit;
    }}
    .markdown img {{
      max-width: 100%;
      height: auto;
      border-radius: var(--radius-xs);
    }}
    .markdown.empty {{
      color: var(--text-muted);
    }}
    .mermaid {{
      text-align: center;
      margin: 1rem 0;
      background: var(--card);
      border-radius: var(--radius-sm);
      padding: 16px;
      border: 1px solid var(--border);
      border: 1px solid var(--border);
    }}

    /* ===== Empty State ===== */
    .empty-state {{
      text-align: center;
      padding: 48px 20px;
      color: var(--text-muted);
    }}
    .empty-state-icon {{
      font-size: 3rem;
      margin-bottom: 12px;
    }}
    .empty-state-text {{
      font-size: 1rem;
    }}

    /* ===== Responsive ===== */
    @media (max-width: 768px) {{
      main {{
        padding: 16px 12px 48px;
      }}
      .page-header {{
        padding: 24px 20px;
        border-radius: var(--radius-sm);
      }}
      .page-header h1 {{
        font-size: 1.35rem;
      }}
      .controls-bar {{
        flex-direction: column;
      }}
      .controls-bar select,
      .controls-bar input {{
        flex: 1 1 100%;
      }}
      .stats-layout {{
        flex-direction: column;
        align-items: stretch;
      }}
      .ring-container {{
        margin: 0 auto;
      }}
      .stats-numbers {{
        grid-template-columns: repeat(2, 1fr);
      }}
      .knowledge-grid {{
        grid-template-columns: 1fr;
      }}
      .question-card {{
        padding: 18px 16px;
      }}
      .question-header {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .option-label {{
        padding: 14px 16px;
      }}
      .btn {{
        padding: 12px 20px;
        font-size: 0.95rem;
        width: 100%;
        justify-content: center;
      }}
      .quiz-nav {{
        flex-direction: column;
      }}
      .quiz-nav .btn {{
        width: 100%;
      }}
    }}
    @media (min-width: 769px) and (max-width: 1024px) {{
      .knowledge-grid {{
        grid-template-columns: repeat(2, 1fr);
      }}
    }}

    /* ===== AI Chat Dialog ===== */
    .ai-chat-btn {{
      display: inline-flex; align-items: center; gap: 4px;
      padding: 6px 14px; font-size: 13px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: #fff; border: none; border-radius: 6px;
      cursor: pointer; transition: opacity .2s, transform .1s;
      float: right; margin-top: 4px;
    }}
    .ai-chat-btn:hover {{ opacity: .9; }}
    .ai-chat-btn:active {{ transform: scale(.97); }}
    .ai-chat-overlay {{
      display: none; position: fixed; inset: 0; z-index: 9999;
      background: rgba(0,0,0,.45); justify-content: center; align-items: center;
    }}
    .ai-chat-overlay.open {{ display: flex; }}
    .ai-chat-panel {{
      width: 520px; max-width: 94vw; height: 680px; max-height: 88vh;
      background: var(--bg); border-radius: 14px; display: flex; flex-direction: column;
      box-shadow: 0 20px 60px rgba(0,0,0,.3); overflow: hidden;
    }}
    .ai-chat-header {{
      display: flex; align-items: center; justify-content: space-between;
      padding: 12px 16px; border-bottom: 1px solid var(--border);
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: #fff;
    }}
    .ai-chat-header h3 {{ margin: 0; font-size: 15px; font-weight: 600; }}
    .ai-chat-header .chat-actions {{ display: flex; gap: 8px; }}
    .ai-chat-header button {{
      background: rgba(255,255,255,.2); border: none; color: #fff;
      border-radius: 6px; padding: 4px 10px; cursor: pointer; font-size: 13px;
    }}
    .ai-chat-header button:hover {{ background: rgba(255,255,255,.35); }}
    .ai-chat-messages {{
      flex: 1; overflow-y: auto; padding: 14px 16px;
      display: flex; flex-direction: column; gap: 10px;
    }}
    .chat-msg {{
      max-width: 88%; padding: 10px 14px; border-radius: 12px;
      font-size: 14px; line-height: 1.6; word-break: break-word;
    }}
    .chat-msg.user {{
      align-self: flex-end; background: #667eea; color: #fff;
      border-bottom-right-radius: 4px;
    }}
    .chat-msg.assistant {{
      align-self: flex-start; background: var(--card); border: 1px solid var(--border);
      border-bottom-left-radius: 4px;
    }}
    .chat-msg.system {{
      align-self: center; background: transparent; color: var(--muted);
      font-size: 12px; text-align: center; padding: 4px 0;
    }}
    .chat-msg.assistant code {{ background: var(--bg-code); padding: 1px 5px; border-radius: 4px; font-size: 13px; }}
    .chat-msg.assistant pre {{
      background: var(--bg-code); padding: 10px; border-radius: 8px;
      overflow-x: auto; margin: 6px 0;
    }}
    .chat-msg.assistant pre code {{ padding: 0; background: none; }}
    .ai-chat-input {{
      display: flex; gap: 8px; padding: 12px 16px; border-top: 1px solid var(--border);
    }}
    .ai-chat-input textarea {{
      flex: 1; border: 1px solid var(--border); border-radius: 8px;
      padding: 8px 12px; font-size: 14px; resize: none;
      background: var(--card); color: var(--fg);
      min-height: 40px; max-height: 120px; font-family: inherit;
    }}
    .ai-chat-input textarea:focus {{ outline: none; border-color: #667eea; }}
    .ai-chat-input button {{
      padding: 8px 18px; background: linear-gradient(135deg, #667eea, #764ba2);
      color: #fff; border: none; border-radius: 8px; cursor: pointer;
      font-size: 14px; font-weight: 500; white-space: nowrap;
    }}
    .ai-chat-input button:disabled {{ opacity: .5; cursor: not-allowed; }}

    /* Chat settings modal */
    .chat-settings-overlay {{
      display: none; position: fixed; inset: 0; z-index: 10001;
      background: rgba(0,0,0,.5); justify-content: center; align-items: center;
    }}
    .chat-settings-overlay.open {{ display: flex; }}
    .chat-settings-box {{
      width: 400px; max-width: 90vw; background: var(--card);
      border-radius: 12px; padding: 24px; box-shadow: 0 12px 40px rgba(0,0,0,.3);
    }}
    .chat-settings-box h3 {{ margin: 0 0 16px; font-size: 16px; }}
    .chat-settings-box label {{ display: block; font-size: 13px; font-weight: 500; margin-bottom: 4px; color: var(--muted); }}
    .chat-settings-box input {{
      width: 100%; padding: 8px 12px; border: 1px solid var(--border);
      border-radius: 8px; font-size: 14px; margin-bottom: 12px;
      background: var(--bg); color: var(--fg);
    }}
    .chat-settings-box input:focus {{ outline: none; border-color: #667eea; }}
    .chat-settings-box .settings-actions {{ display: flex; gap: 10px; justify-content: flex-end; margin-top: 6px; }}
    .chat-settings-box .settings-actions button {{
      padding: 8px 20px; border-radius: 8px; border: none; cursor: pointer; font-size: 14px;
    }}
    .chat-settings-box .btn-save {{ background: #667eea; color: #fff; }}
    .chat-settings-box .btn-cancel {{ background: var(--border); color: var(--fg); }}

    .typing-indicator {{ display: inline-flex; gap: 4px; padding: 4px 0; }}
    .typing-indicator span {{
      width: 6px; height: 6px; background: var(--muted); border-radius: 50%;
      animation: typing .8s infinite alternate;
    }}
    .typing-indicator span:nth-child(2) {{ animation-delay: .2s; }}
    .typing-indicator span:nth-child(3) {{ animation-delay: .4s; }}
    @keyframes typing {{ to {{ opacity: .3; transform: translateY(-3px); }} }}
  </style>
</head>
<body>
  <main>
    <header class="page-header">
      <button class="theme-toggle" id="themeToggle" aria-label="切换主题">🌙 暗色</button>
      <h1>📝 {site_data['meta'].get('paper_name', '刷题笔记')}</h1>
      <p class="meta-line">共 {site_data['meta'].get('item_count')} 题 · 生成于 {site_data['generated_at']}</p>
    </header>

    <div class="controls-bar">
      <select id="knowledgeFilter" aria-label="筛选知识点"></select>
      <input id="searchInput" type="search" placeholder="🔍 搜索题干 / 解析 / AI 笔记" aria-label="搜索题目" />
    </div>

    <section class="stats-card" aria-label="答题统计">
      <h2>📊 答题统计</h2>
      <div id="statsContainer">
        <div class="stats-layout">
          <div class="ring-container">
            <svg width="120" height="120" viewBox="0 0 120 120">
              <circle cx="60" cy="60" r="52" fill="none" stroke="#2d2d3f" stroke-width="10"/>
              <circle cx="60" cy="60" r="52" fill="none" stroke="var(--accent)" stroke-width="10"
                stroke-dasharray="326.73" stroke-dashoffset="326.73" stroke-linecap="round" id="ringProgress"/>
            </svg>
            <div class="ring-label">
              <span class="ring-pct" id="ringPct">0%</span>
              <span class="ring-text">正确率</span>
            </div>
          </div>
          <div class="stats-numbers">
            <div class="sn-item"><div class="sn-val accent" id="snTotal">0</div><div class="sn-lbl">总题数</div></div>
            <div class="sn-item"><div class="sn-val accent" id="snAnswered">0</div><div class="sn-lbl">已答题</div></div>
            <div class="sn-item"><div class="sn-val green" id="snCorrect">0</div><div class="sn-lbl">答对</div></div>
            <div class="sn-item"><div class="sn-val red" id="snIncorrect">0</div><div class="sn-lbl">答错</div></div>
          </div>
        </div>
        <div class="stats-progress">
          <div class="stats-progress-bar"><div class="stats-progress-fill" id="statsProgressFill" style="width:0%"></div></div>
          <div class="stats-progress-text" id="statsProgressText">进度 0%</div>
        </div>
      </div>
    </section>

    <section aria-label="知识点总结">
      <h2 class="section-title">💡 知识点总结</h2>
      <div id="knowledgeContainer" class="knowledge-grid">{knowledge_initial}</div>
    </section>

    <section aria-label="题目列表">
      <h2 class="section-title">📋 题目列表</h2>
      <div id="questionsContainer">{questions_initial}</div>
    </section>
  </main>
  <script>
    const DATA = {data_json};

    // ===== Theme toggle =====
    (function() {{
      const THEME_KEY = 'quiz_theme';
      const root = document.documentElement;
      const saved = localStorage.getItem(THEME_KEY);
      if (saved) root.setAttribute('data-theme', saved);
      function updateBtn() {{
        const btn = document.getElementById('themeToggle');
        if (!btn) return;
        const isLight = root.getAttribute('data-theme') === 'light';
        btn.textContent = isLight ? '🌙 暗色' : '☀️ 亮色';
        btn.setAttribute('aria-label', isLight ? '切换到暗色模式' : '切换到亮色模式');
      }}
      document.addEventListener('DOMContentLoaded', () => {{
        updateBtn();
        document.getElementById('themeToggle').addEventListener('click', () => {{
          const isLight = root.getAttribute('data-theme') === 'light';
          const next = isLight ? 'dark' : 'light';
          root.setAttribute('data-theme', next);
          localStorage.setItem(THEME_KEY, next);
          updateBtn();
        }});
      }});
    }})();

    function formatPrompt(text) {{
      if (!text) return '';
      const escaped = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      return escaped
        .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" style="max-width: 100%; height: auto;" />')
        .replace(/!(https?:\/\/[^\s<>"\'\\[\]]+)/g, '<img src="$1" alt="图片" style="max-width: 100%; height: auto;" />');
    }}

    function renderBasicMarkdown(raw = '') {{
      if (!raw) return '<p>（尚未生成，稍后重试）</p>';
      const normalized = raw.split('\\r\\n').join('\\n').split('\\r').join('\\n');
      const lines = normalized.split('\\n');
      const htmlParts = [];
      let inUl = false;
      let inOl = false;
      const closeLists = () => {{
        if (inUl) {{ htmlParts.push('</ul>'); inUl = false; }}
        if (inOl) {{ htmlParts.push('</ol>'); inOl = false; }}
      }};
      const formatInline = (text) => {{
        const escaped = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        return escaped
          .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" style="max-width: 100%; height: auto;" />')
          .replace(/!(https?:\/\/[^\s<>"\'\\[\]]+(?:\.[a-zA-Z]{{2,}}|\/[^\s<>"\'\\[\]]*)?)/g, '<img src="$1" alt="图片" style="max-width: 100%; height: auto;" />')
          .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
          .replace(/__(.+?)__/g, '<strong>$1</strong>')
          .replace(/`([^`]+)`/g, '<code>$1</code>')
          .replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>')
          .replace(/(?<!_)_(?!_)(.+?)(?<!_)_(?!_)/g, '<em>$1</em>');
      }};
      const renderTable = (rows) => {{
        if (!rows.length) return '';
        const splitRow = (row) => row.replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => formatInline(cell.trim()));
        const headerCells = splitRow(rows[0]);
        let alignments = Array(headerCells.length).fill('left');
        let bodyRows = rows.slice(1);
        if (bodyRows.length) {{
          const alignParts = bodyRows[0].replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim());
          const isDivider = alignParts.length === headerCells.length && alignParts.every((cell) => /^:?-{{3,}}:?$/.test(cell));
          if (isDivider) {{
            alignments = alignParts.map((cell) => {{
              const left = cell.startsWith(':');
              const right = cell.endsWith(':');
              if (left && right) return 'center';
              if (right) return 'right';
              return 'left';
            }});
            bodyRows = bodyRows.slice(1);
          }}
        }}
        const headerHtml = '<tr>' + headerCells.map((cell, idx) => `<th style="text-align: ${{alignments[idx] || 'left'}}">${{cell}}</th>`).join('') + '</tr>';
        const bodyHtml = bodyRows.map((row) => {{
          const cells = splitRow(row);
          return '<tr>' + cells.map((cell, idx) => `<td style="text-align: ${{alignments[idx] || 'left'}}">${{cell}}</td>`).join('') + '</tr>';
        }}).join('');
        const filler = '<tr>' + headerCells.map((_, idx) => `<td style="text-align: ${{alignments[idx] || 'left'}}">&nbsp;</td>`).join('') + '</tr>';
        return `<table><thead>${{headerHtml}}</thead><tbody>${{bodyHtml || filler}}</tbody></table>`;
      }};
      for (let i = 0; i < lines.length; i += 1) {{
        const trimmed = lines[i].trim();
        if (!trimmed) {{ closeLists(); continue; }}
        const headingMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
        if (headingMatch) {{
          closeLists();
          const level = headingMatch[1].length + 1; // h2-h5 (h1 reserved for page)
          const tag = 'h' + Math.min(level, 5);
          htmlParts.push(`<${{tag}} class="md-heading">${{formatInline(headingMatch[2])}}</${{tag}}>`);
          continue;
        }}
        if (trimmed === '---' || trimmed === '***' || trimmed === '___') {{
          closeLists();
          htmlParts.push('<hr class="md-hr" />');
          continue;
        }}
        if (/^\|.+\|$/.test(trimmed)) {{
          closeLists();
          const tableRows = [];
          let j = i;
          while (j < lines.length && /^\|.+\|$/.test(lines[j].trim())) {{
            tableRows.push(lines[j].trim());
            j += 1;
          }}
          if (tableRows.length) {{ htmlParts.push(renderTable(tableRows)); }}
          i = j - 1;
          continue;
        }}
        if (/^\d+\.\s+/.test(trimmed)) {{
          if (!inOl) {{ closeLists(); htmlParts.push('<ol>'); inOl = true; }}
          htmlParts.push('<li>' + formatInline(trimmed.replace(/^\d+\.\s+/, '')) + '</li>');
          continue;
        }}
        if (/^[-*+]\s+/.test(trimmed)) {{
          if (!inUl) {{ closeLists(); htmlParts.push('<ul>'); inUl = true; }}
          htmlParts.push('<li>' + formatInline(trimmed.slice(1).trim()) + '</li>');
          continue;
        }}
        closeLists();
        htmlParts.push('<p>' + formatInline(trimmed) + '</p>');
      }}
      closeLists();
      const joined = htmlParts.filter(Boolean).join('');
      return joined || '<p>（尚未生成，稍后重试）</p>';
    }}

    const md = window.markdownit
      ? window.markdownit({{html: true, linkify: true, breaks: true}})
      : {{ render: (raw = '') => renderBasicMarkdown(raw) }};
    if (md && typeof md.enable === 'function') {{
      md.enable('table');
      md.enable('strikethrough');
    }}

    const STORAGE_KEY = 'quiz_progress_' + btoa(unescape(encodeURIComponent(document.title))).slice(0, 32);

    const state = {{
      knowledge: 'all',
      search: '',
      currentQuestionIndex: 0,
      quizMode: true,
      stats: {{ total: 0, answered: 0, correct: 0, incorrect: 0 }},
      answeredQuestions: {{}}
    }};

    function saveState() {{
      try {{
        localStorage.setItem(STORAGE_KEY, JSON.stringify({{
          currentQuestionIndex: state.currentQuestionIndex,
          quizMode: state.quizMode,
          knowledge: state.knowledge,
          answeredQuestions: state.answeredQuestions,
          scrollY: window.scrollY
        }}));
      }} catch(e) {{}}
    }}

    function loadState() {{
      try {{
        const saved = localStorage.getItem(STORAGE_KEY);
        if (!saved) return;
        const p = JSON.parse(saved);
        if (typeof p.currentQuestionIndex === 'number') state.currentQuestionIndex = p.currentQuestionIndex;
        if (typeof p.quizMode === 'boolean') state.quizMode = p.quizMode;
        if (p.knowledge) state.knowledge = p.knowledge;
        if (p.answeredQuestions) state.answeredQuestions = p.answeredQuestions;
        if (typeof p.scrollY === 'number') state._savedScrollY = p.scrollY;
      }} catch(e) {{}}
    }}

    function restoreAnsweredState(questionId) {{
      const saved = state.answeredQuestions[questionId];
      if (!saved) return;
      const questionDiv = document.querySelector(`[data-question-id="${{questionId}}"]`);
      if (!questionDiv) return;
      const correctAnswers = questionDiv.dataset.correctAnswers.split(',').filter(Boolean);
      const inputs = questionDiv.querySelectorAll('input[type="radio"], input[type="checkbox"]');
      inputs.forEach(input => {{
        if (saved.selected.includes(input.value)) input.checked = true;
        input.disabled = true;
        const optionItem = input.closest('.option-item');
        if (input.checked) optionItem.classList.add('selected');
        if (correctAnswers.includes(input.value)) optionItem.classList.add('correct');
        else if (input.checked) optionItem.classList.add('incorrect');
      }});
      const feedbackDiv = questionDiv.querySelector('.answer-feedback');
      if (feedbackDiv) {{
        feedbackDiv.style.display = 'block';
        feedbackDiv.className = `answer-feedback ${{saved.isCorrect ? 'correct' : 'incorrect'}}`;
        feedbackDiv.textContent = saved.isCorrect ? '✓ 回答正确！' : '✗ 回答错误';
      }}
      const submitBtn = questionDiv.querySelector('.submit-btn');
      if (submitBtn) submitBtn.style.display = 'none';
      if (!saved.isCorrect) {{
        const correctAnswerP = questionDiv.querySelector('.correct-answer');
        if (correctAnswerP) correctAnswerP.style.display = 'block';
        const analysisDetails = questionDiv.querySelector('.answer-analysis');
        if (analysisDetails) {{ analysisDetails.style.display = 'block'; analysisDetails.open = true; }}
      }}
    }}

    function renderMarkdownWithMermaid(content) {{
      if (!content) return '<p>（尚未生成，稍后重试）</p>';
      const mermaidRegex = /```mermaid\\n([\s\S]*?)\\n```/g;
      let processed = content.replace(mermaidRegex, (match, graphDef) => `<div class="mermaid">${{graphDef.trim()}}</div>`);
      return md.render ? md.render(processed) : renderBasicMarkdown(processed);
    }}
    function renderKnowledge() {{
      const container = document.getElementById('knowledgeContainer');
      container.innerHTML = '';
      DATA.knowledge_points.forEach((kp) => {{
        const name = kp.name;
        const count = (kp.related_questions || []).length;
        const summaryMd = typeof kp.summary_markdown === 'object' ? (kp.summary_markdown.response || '') : (kp.summary_markdown || '');
        const card = document.createElement('article');
        card.className = 'knowledge-card' + (state.knowledge === name ? ' active' : '');
        const summaryHTML = summaryMd && summaryMd.trim() ? renderMarkdownWithMermaid(summaryMd) : '<p class="empty-text" style="color:var(--text-muted)">（尚未生成，稍后重试）</p>';
        const summaryBlock = summaryMd && summaryMd.trim()
          ? `<details class="knowledge-details"><summary>查看大模型总结</summary><div class="markdown">${{summaryHTML}}</div></details>`
          : `<div class="markdown empty">${{summaryHTML}}</div>`;
        card.innerHTML = `
          <div class="knowledge-header">
            <h3>${{name}}</h3>
            <span class="badge badge-accent kp-count-btn" data-kp="${{name}}" title="点击刷此知识点题库">共 ${{count}} 题 ▶</span>
          </div>
          ${{summaryBlock}}
        `;
        // 整张卡片点击切换知识点
        card.style.cursor = 'pointer';
        card.addEventListener('click', (e) => {{
          // 如果点击的是 details/summary 展开操作，不干预
          if (e.target.closest('details') || e.target.closest('summary')) return;
          jumpToKnowledge(name);
        }});
        // badge 单独处理（阻止冒泡避免双触发）
        const badge = card.querySelector('.kp-count-btn');
        if (badge) {{
          badge.addEventListener('click', (e) => {{
            e.stopPropagation();
            jumpToKnowledge(name);
          }});
        }}
        container.append(card);
      }});
    }}

    function jumpToKnowledge(name) {{
      state.knowledge = name;
      state.currentQuestionIndex = 0;
      saveState();
      // 同步下拉框
      const sel = document.getElementById('knowledgeFilter');
      if (sel) sel.value = name;
      renderKnowledge();
      renderQuestions();
      // 滚动到题目区
      const qSection = document.getElementById('questionsContainer');
      if (qSection) {{
        setTimeout(() => qSection.scrollIntoView({{ behavior: 'smooth', block: 'start' }}), 80);
      }}
      showToast(`已切换到：${{name}}`, 'info');
    }}

    function filterQuestions() {{
      return DATA.questions.filter((q) => {{
        const matchKP = state.knowledge === 'all' || q.knowledge_point === state.knowledge;
        if (!matchKP) return false;
        if (!state.search) return true;
        const haystack = [
          q.prompt, q.knowledge_point, q.model_response || '',
          ...(q.sub_questions || []).map((sub) => [sub.official_analysis || '', (sub.options || []).map((o) => o.text).join(' ')].join(' '))
        ].join(' ').toLowerCase();
        return haystack.includes(state.search.toLowerCase());
      }});
    }}

    function buildSubQuestionHTML(sub, subIdx, q) {{
      const correctCount = (sub.options || []).filter(o => o.is_correct).length;
      const isMultiple = correctCount > 1;
      const inputType = isMultiple ? 'checkbox' : 'radio';
      const questionId = `q${{q.id || q.index}}_${{subIdx}}`;
      const answers = sub.correct_letters || [];
      const options = (sub.options || []).map((opt) => {{
        const optionId = `${{questionId}}_${{opt.label}}`;
        return `<li class="option-item"><label for="${{optionId}}" class="option-label"><input type="${{inputType}}" id="${{optionId}}" name="${{questionId}}" value="${{opt.label}}" class="option-input" /><span class="option-text">${{opt.label}}. ${{opt.text}}</span></label></li>`;
      }}).join('');
      const subQuestionText = sub.question ? `<div class="sub-question-text">${{md.render(sub.question)}}</div>` : '';
      const correctAnswerDetail = sub.correct_answer
        ? `<details class="rich-details answer-analysis" style="display:none;"><summary>参考答案</summary><div class="markdown">${{md.render(sub.correct_answer)}}</div></details>` : '';
      const answersP = answers.length ? `<p class="correct-answer" style="display:none;"><strong>参考答案：</strong>${{answers.join('、')}}</p>` : '';
      const official = sub.official_analysis
        ? `<details class="rich-details answer-analysis" style="display:none;"><summary>官方解析</summary><div class="markdown">${{md.render(sub.official_analysis)}}</div></details>` : '';
      const hasOptions = sub.options && sub.options.length > 0;
      const interaction = hasOptions
        ? `<ul class="options interactive-options">${{options}}</ul>
           <div class="question-actions">
             <button class="btn btn-primary submit-btn" onclick="submitAnswer('${{questionId}}')">提交答案</button>
             <button class="btn btn-warning show-answer-btn" onclick="showAnswer('${{questionId}}')" style="display:none;">查看答案</button>
           </div>`
        : `<div class="question-actions">
             <button class="btn btn-warning show-answer-btn" onclick="showAnswer('${{questionId}}')" style="display:inline-flex;">查看答案</button>
           </div>`;
      return `<div class="sub-question" data-question-id="${{questionId}}" data-correct-answers="${{answers.join(',')}}" data-is-multiple="${{isMultiple}}">
        <h4>子题${{sub.label}}</h4>
        ${{subQuestionText}}
        ${{interaction}}
        <div class="answer-feedback" style="display:none;"></div>
        ${{answersP}}
        ${{correctAnswerDetail}}
        ${{official}}
      </div>`;
    }}

    function buildAiNote(q) {{
      const resp = typeof q.model_response === 'object' ? (q.model_response.response || '') : (q.model_response || '');
      if (resp && resp.trim()) {{
        return `<details class="rich-details"><summary>AI 记忆笔记</summary><div class="markdown">${{renderMarkdownWithMermaid(resp)}}</div></details>`;
      }}
      return '<details class="rich-details"><summary>AI 记忆笔记</summary><div class="markdown empty">暂无内容</div></details>';
    }}

    function renderQuestions() {{
      const container = document.getElementById('questionsContainer');
      container.innerHTML = '';
      const results = filterQuestions();
      if (!results.length) {{
        container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">🔍</div><div class="empty-state-text">未找到匹配的题目，试试其他关键词？</div></div>';
        return;
      }}
      updateStats(results);
      if (state.quizMode) {{
        renderSingleQuestion(results);
      }} else {{
        renderAllQuestions(results);
      }}
      Object.keys(state.answeredQuestions).forEach(qId => restoreAnsweredState(qId));
      updateStats(results);
      // Re-render math formulas after DOM update
      setTimeout(renderAllMath, 100);
    }}

    function renderSingleQuestion(questions) {{
      const container = document.getElementById('questionsContainer');
      const cur = questions[state.currentQuestionIndex];
      if (!cur) {{
        container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">🎉</div><div class="empty-state-text">没有更多题目了</div></div>';
        return;
      }}
      const pct = ((state.currentQuestionIndex + 1) / questions.length * 100).toFixed(1);
      const progressBar = `<div class="quiz-progress-bar">
        <div class="qp-top">
          <span class="qp-label">第 ${{state.currentQuestionIndex + 1}} 题 / 共 ${{questions.length}} 题</span>
          <span class="qp-count">${{pct}}%</span>
        </div>
        <div class="qp-bar"><div class="qp-fill" style="width:${{pct}}%"></div></div>
      </div>`;
      const diffBadge = cur.difficulty ? `<span class="badge badge-green">难度 ${{cur.difficulty}}</span>` : '';
      const card = document.createElement('div');
      card.innerHTML = progressBar;
      const qCard = document.createElement('article');
      qCard.className = 'question-card';
      const subs = (cur.sub_questions || []).map((sub, idx) => buildSubQuestionHTML(sub, idx, cur)).join('');
      const isAnswered = Object.keys(state.answeredQuestions).some(k => k.startsWith(`q${{cur.id || cur.index}}_`));
      const nav = `<div class="quiz-nav">
        <button class="btn btn-outline" onclick="previousQuestion()" ${{state.currentQuestionIndex === 0 ? 'disabled' : ''}} title="快捷键 ←">← 上一题</button>
        <button class="btn btn-ghost" onclick="toggleQuizMode()">📋 全部题目</button>
        <button class="btn btn-outline" onclick="nextQuestion()" ${{state.currentQuestionIndex >= questions.length - 1 ? 'disabled' : ''}} title="快捷键 →">下一题 →</button>
      </div>`;
      qCard.innerHTML = `
        <div class="question-header">
          <h3>第${{cur.index}}题（题号：${{cur.sort || cur.index}}）</h3>
          <span class="badge badge-accent">${{cur.knowledge_point}}</span>
          ${{diffBadge}}
          <button class="ai-chat-btn" onclick="openAiChat(${{cur.index}})">🤖 AI对话</button>
        </div>
        <p class="question-prompt">${{formatPrompt(cur.prompt)}}</p>
        ${{subs}}
        ${{nav}}
        ${{buildAiNote(cur)}}
      `;
      card.appendChild(qCard);
      container.appendChild(card);
    }}

    function renderAllQuestions(results) {{
      const container = document.getElementById('questionsContainer');
      const toggle = document.createElement('div');
      toggle.className = 'quiz-mode-toggle';
      toggle.innerHTML = '<button class="btn btn-success" onclick="toggleQuizMode()">🎯 单题模式</button>';
      container.appendChild(toggle);
      results.forEach((q) => {{
        const card = document.createElement('article');
        card.className = 'question-card';
        const diffBadge = q.difficulty ? `<span class="badge badge-green">难度 ${{q.difficulty}}</span>` : '';
        const subs = (q.sub_questions || []).map((sub, idx) => buildSubQuestionHTML(sub, idx, q)).join('');
        card.innerHTML = `
          <div class="question-header">
            <h3>第${{q.index}}题（题号：${{q.sort || q.index}}）</h3>
            <span class="badge badge-accent">${{q.knowledge_point}}</span>
            ${{diffBadge}}
            <button class="ai-chat-btn" onclick="openAiChat(${{q.index}})">🤖 AI对话</button>
          </div>
          <p class="question-prompt">${{formatPrompt(q.prompt)}}</p>
          ${{subs}}
          ${{buildAiNote(q)}}
        `;
        container.appendChild(card);
      }});
    }}
    function populateFilter() {{
      const select = document.getElementById('knowledgeFilter');
      const options = ['<option value="all">全部知识点</option>'];
      DATA.knowledge_points.forEach((kp) => {{
        options.push(`<option value="${{kp.name}}">${{kp.name}}（${{kp.related_questions.length}}）</option>`);
      }});
      select.innerHTML = options.join('');
      if (state.knowledge !== 'all') select.value = state.knowledge;
      select.addEventListener('change', (e) => {{
        state.knowledge = e.target.value;
        state.currentQuestionIndex = 0;
        saveState();
        renderKnowledge();
        renderQuestions();
      }});
    }}

    function registerSearch() {{
      const input = document.getElementById('searchInput');
      let timer;
      input.addEventListener('input', (e) => {{
        clearTimeout(timer);
        timer = setTimeout(() => {{
          state.search = e.target.value.trim();
          state.currentQuestionIndex = 0;
          renderQuestions();
        }}, 200);
      }});
    }}

    function initMermaid() {{
      if (window.mermaid) {{
        mermaid.initialize({{
          startOnLoad: false,
          theme: 'dark',
          themeVariables: {{
            primaryColor: '#7c6ff7',
            primaryTextColor: '#e2e8f0',
            primaryBorderColor: '#a89cf8',
            lineColor: '#94a3b8',
            secondaryColor: '#1c1c28',
            tertiaryColor: '#16161f',
            background: '#0a0a0f',
            mainBkg: '#16161f',
            nodeBorder: '#7c6ff7'
          }}
        }});
      }}
    }}

    function renderMermaidCharts() {{
      if (window.mermaid) {{
        const els = document.querySelectorAll('.mermaid');
        els.forEach((el, index) => {{
          const def = el.textContent;
          const id = `mermaid-${{Date.now()}}-${{index}}`;
          el.id = id;
          mermaid.render(id + '-svg', def).then((result) => {{
            el.innerHTML = result.svg;
          }}).catch(() => {{
            el.innerHTML = '<p style="color:var(--red);">图表渲染失败</p>';
          }});
        }});
      }}
    }}

    function updateStats(questions) {{
      const totalSub = questions.reduce((s, q) => s + (q.sub_questions || []).length, 0);
      const answered = Object.keys(state.answeredQuestions).length;
      const correct = Object.values(state.answeredQuestions).filter(a => a.isCorrect).length;
      const incorrect = answered - correct;
      state.stats = {{ total: totalSub, answered, correct, incorrect }};
      renderStatsDisplay();
    }}

    function renderStatsDisplay() {{
      const {{ total, answered, correct, incorrect }} = state.stats;
      const accuracy = answered > 0 ? Math.round((correct / answered) * 100) : 0;
      const progress = total > 0 ? Math.round((answered / total) * 100) : 0;
      const circumference = 2 * Math.PI * 52;
      const offset = circumference - (accuracy / 100) * circumference;

      const ring = document.getElementById('ringProgress');
      if (ring) {{
        ring.style.strokeDasharray = circumference;
        ring.style.strokeDashoffset = offset;
        ring.style.transition = 'stroke-dashoffset 0.6s ease';
      }}
      const ringPct = document.getElementById('ringPct');
      if (ringPct) ringPct.textContent = accuracy + '%';

      const snTotal = document.getElementById('snTotal');
      const snAnswered = document.getElementById('snAnswered');
      const snCorrect = document.getElementById('snCorrect');
      const snIncorrect = document.getElementById('snIncorrect');
      if (snTotal) snTotal.textContent = total;
      if (snAnswered) snAnswered.textContent = answered;
      if (snCorrect) snCorrect.textContent = correct;
      if (snIncorrect) snIncorrect.textContent = incorrect;

      const fill = document.getElementById('statsProgressFill');
      const txt = document.getElementById('statsProgressText');
      if (fill) fill.style.width = progress + '%';
      if (txt) txt.textContent = `进度 ${{progress}}%（${{answered}}/${{total}}）`;
    }}

    function submitAnswer(questionId) {{
      const questionDiv = document.querySelector(`[data-question-id="${{questionId}}"]`);
      if (!questionDiv) return;
      const correctAnswers = questionDiv.dataset.correctAnswers.split(',').filter(Boolean);
      const inputs = questionDiv.querySelectorAll('input[type="radio"], input[type="checkbox"]');
      const selectedAnswers = Array.from(inputs).filter(i => i.checked).map(i => i.value);
      if (selectedAnswers.length === 0) {{
        showToast('请先选择答案', 'warn');
        const ul = questionDiv.querySelector('.options');
        if (ul) {{ ul.style.animation = 'shake 0.4s ease'; setTimeout(() => ul.style.animation = '', 500); }}
        return;
      }}
      inputs.forEach(input => input.disabled = true);
      const isCorrect = correctAnswers.length === selectedAnswers.length &&
                       correctAnswers.every(a => selectedAnswers.includes(a));
      const feedbackDiv = questionDiv.querySelector('.answer-feedback');
      feedbackDiv.style.display = 'block';
      feedbackDiv.className = `answer-feedback ${{isCorrect ? 'correct' : 'incorrect'}}`;
      feedbackDiv.innerHTML = isCorrect
        ? '<span class="feedback-icon">✓</span><span class="feedback-msg">回答正确！继续加油</span>'
        : '<span class="feedback-icon">✗</span><span class="feedback-msg">回答错误，看看解析吧</span>';
      inputs.forEach(input => {{
        const optionItem = input.closest('.option-item');
        if (input.checked) optionItem.classList.add('selected');
        if (correctAnswers.includes(input.value)) optionItem.classList.add('correct');
        else if (input.checked) optionItem.classList.add('incorrect');
      }});
      const submitBtn = questionDiv.querySelector('.submit-btn');
      const showAnswerBtn = questionDiv.querySelector('.show-answer-btn');
      submitBtn.style.display = 'none';
      showAnswerBtn.style.display = 'inline-flex';
      state.answeredQuestions[questionId] = {{ selected: selectedAnswers, isCorrect }};
      saveState();
      if (!isCorrect) showAnswer(questionId);
      const results = filterQuestions();
      updateStats(results);
      // 平滑滚动到反馈区
      setTimeout(() => {{
        feedbackDiv.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
      }}, 100);
    }}

    function showAnswer(questionId) {{
      const questionDiv = document.querySelector(`[data-question-id="${{questionId}}"]`);
      if (!questionDiv) return;
      const correctAnswerP = questionDiv.querySelector('.correct-answer');
      if (correctAnswerP) correctAnswerP.style.display = 'block';
      const analysisDetailsList = questionDiv.querySelectorAll('.answer-analysis');
      analysisDetailsList.forEach(d => {{ d.style.display = 'block'; d.open = true; }});
      const showAnswerBtn = questionDiv.querySelector('.show-answer-btn');
      if (showAnswerBtn) showAnswerBtn.style.display = 'none';
    }}

    function scrollToQuestion() {{
      const el = document.getElementById('questionsContainer') || document.querySelector('.question-card');
      if (el) el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }}

    function previousQuestion() {{
      if (state.currentQuestionIndex > 0) {{
        state.currentQuestionIndex--;
        saveState();
        renderQuestions();
        scrollToQuestion();
        animateCardIn();
      }}
    }}

    function nextQuestion() {{
      const questions = filterQuestions();
      if (state.currentQuestionIndex < questions.length - 1) {{
        state.currentQuestionIndex++;
        saveState();
        renderQuestions();
        scrollToQuestion();
        animateCardIn();
      }}
    }}

    function animateCardIn() {{
      setTimeout(() => {{
        const card = document.querySelector('.question-card');
        if (card) {{
          card.style.opacity = '0';
          card.style.transform = 'translateY(12px)';
          card.style.transition = 'none';
          requestAnimationFrame(() => {{
            requestAnimationFrame(() => {{
              card.style.transition = 'opacity 0.25s ease, transform 0.25s ease';
              card.style.opacity = '1';
              card.style.transform = 'translateY(0)';
            }});
          }});
        }}
      }}, 10);
    }}

    let _toastTimer;
    function showToast(msg, type = 'info') {{
      let toast = document.getElementById('_toast');
      if (!toast) {{
        toast = document.createElement('div');
        toast.id = '_toast';
        document.body.appendChild(toast);
      }}
      toast.textContent = msg;
      toast.className = '_toast-show _toast-' + type;
      clearTimeout(_toastTimer);
      _toastTimer = setTimeout(() => {{ toast.className = ''; }}, 2200);
    }}

    function toggleQuizMode() {{
      state.quizMode = !state.quizMode;
      if (state.quizMode) state.currentQuestionIndex = 0;
      saveState();
      renderQuestions();
    }}

    document.addEventListener('DOMContentLoaded', () => {{
      initMermaid();
      loadState();
      populateFilter();
      registerSearch();
      renderKnowledge();
      renderQuestions();
      // Render math formulas after initial load
      setTimeout(renderAllMath, 500);
      // Expand knowledge card to full width when details opens
      document.getElementById('knowledgeContainer').addEventListener('toggle', (e) => {{
        if (e.target.classList.contains('knowledge-details')) {{
          const card = e.target.closest('.knowledge-card');
          if (card) card.classList.toggle('expanded', e.target.open);
        }}
      }}, true);
      Object.keys(state.answeredQuestions).forEach(qId => restoreAnsweredState(qId));
      const results = filterQuestions();
      updateStats(results);
      if (state._savedScrollY) {{
        setTimeout(() => window.scrollTo(0, state._savedScrollY), 150);
      }}
      let scrollTimer = null;
      window.addEventListener('scroll', () => {{
        if (scrollTimer) clearTimeout(scrollTimer);
        scrollTimer = setTimeout(() => saveState(), 300);
      }});
      window.addEventListener('beforeunload', () => saveState());
      // 键盘左右键翻页
      document.addEventListener('keydown', (e) => {{
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        if (!state.quizMode) return;
        if (e.key === 'ArrowRight' || e.key === 'l') nextQuestion();
        if (e.key === 'ArrowLeft'  || e.key === 'h') previousQuestion();
      }});
      setTimeout(() => renderMermaidCharts(), 100);
    }});

    // ===== AI Chat System =====
    const CHAT_SETTINGS_KEY = 'ai_chat_llm_settings';

    function getChatSettings() {{
      try {{
        return JSON.parse(localStorage.getItem(CHAT_SETTINGS_KEY)) || {{}};
      }} catch {{ return {{}}; }}
    }}
    function saveChatSettings(s) {{
      localStorage.setItem(CHAT_SETTINGS_KEY, JSON.stringify(s));
    }}

    // Chat state
    const chatState = {{
      messages: [],   // {{role, content}}
      questionCtx: '',
      loading: false
    }};

    function buildQuestionContext(q) {{
      let ctx = '【题目】\\n' + (q.prompt || '') + '\\n\\n';
      if (q.sub_questions && q.sub_questions.length) {{
        q.sub_questions.forEach((sub, i) => {{
          if (sub.question) ctx += '【子题' + sub.label + '】' + sub.question + '\\n';
          if (sub.options && sub.options.length) {{
            sub.options.forEach(opt => {{
              ctx += opt.label + '. ' + opt.text + (opt.is_correct ? ' ✅' : '') + '\\n';
            }});
          }}
          if (sub.correct_letters && sub.correct_letters.length)
            ctx += '正确答案：' + sub.correct_letters.join(', ') + '\\n';
          if (sub.correct_answer) ctx += '参考答案：' + sub.correct_answer + '\\n';
          ctx += '\\n';
        }});
      }}
      // Append existing AI analysis
      const resp = typeof q.model_response === 'object' ? (q.model_response.response || '') : (q.model_response || '');
      if (resp && resp.trim()) {{
        ctx += '【已有的AI解析/笔记】\\n' + resp.trim() + '\\n';
      }}
      return ctx;
    }}

    function findQuestionByIdx(idx) {{
      return DATA.questions.find(q => q.index === idx || q.id === idx);
    }}

    function openAiChat(qIndex) {{
      const q = findQuestionByIdx(qIndex);
      if (!q) return;
      chatState.messages = [];
      chatState.questionCtx = buildQuestionContext(q);
      chatState.loading = false;

      const overlay = document.getElementById('aiChatOverlay');
      const msgs = document.getElementById('aiChatMessages');
      const title = document.getElementById('aiChatTitle');
      title.textContent = 'AI 对话 · 第' + q.index + '题';
      msgs.innerHTML = '';
      overlay.classList.add('open');

      // System message with context
      chatState.messages.push({{
        role: 'system',
        content: '你是一个耐心的软考备考辅导老师。学生会问你关于具体题目的疑问，请用通俗易懂的方式讲解。以下是对话相关的题目信息：\\n\\n' + chatState.questionCtx
      }});

      appendChatMsg('assistant', '你好！我看到了这道题的内容。有什么不理解的地方，直接问我吧 👋');
      document.getElementById('aiChatInput').focus();
    }}

    function closeAiChat() {{
      document.getElementById('aiChatOverlay').classList.remove('open');
    }}

    function appendChatMsg(role, content) {{
      const msgs = document.getElementById('aiChatMessages');
      const div = document.createElement('div');
      div.className = 'chat-msg ' + role;
      // Simple markdown rendering for assistant messages
      if (role === 'assistant') {{
        div.innerHTML = renderSimpleMd(content);
      }} else {{
        div.textContent = content;
      }}
      msgs.appendChild(div);
      msgs.scrollTop = msgs.scrollHeight;
    }}

    function renderSimpleMd(text) {{
      return text
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
        .replace(/\\n/g, '<br>');
    }}

    function showTyping() {{
      const msgs = document.getElementById('aiChatMessages');
      const div = document.createElement('div');
      div.className = 'chat-msg assistant';
      div.id = 'typingIndicator';
      div.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
      msgs.appendChild(div);
      msgs.scrollTop = msgs.scrollHeight;
    }}

    function hideTyping() {{
      const el = document.getElementById('typingIndicator');
      if (el) el.remove();
    }}

    async function sendChatMessage() {{
      const input = document.getElementById('aiChatInput');
      const text = input.value.trim();
      if (!text || chatState.loading) return;

      const settings = getChatSettings();
      if (!settings.baseUrl || !settings.model || !settings.apiKey) {{
        openChatSettings();
        return;
      }}

      chatState.messages.push({{ role: 'user', content: text }});
      appendChatMsg('user', text);
      input.value = '';
      input.style.height = 'auto';
      chatState.loading = true;
      document.getElementById('aiChatSendBtn').disabled = true;
      showTyping();

      try {{
        const apiUrl = settings.baseUrl.replace(/\\/+$/, '') + '/v1/chat/completions';
        const res = await fetch(apiUrl, {{
          method: 'POST',
          headers: {{
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + settings.apiKey
          }},
          body: JSON.stringify({{
            model: settings.model,
            messages: chatState.messages,
            stream: false
          }})
        }});

        if (!res.ok) {{
          const err = await res.text();
          throw new Error('API错误 (' + res.status + '): ' + err.substring(0, 200));
        }}

        const data = await res.json();
        const reply = data.choices?.[0]?.message?.content || '（无回复内容）';
        chatState.messages.push({{ role: 'assistant', content: reply }});
        hideTyping();
        appendChatMsg('assistant', reply);
      }} catch(e) {{
        hideTyping();
        appendChatMsg('assistant', '❌ ' + e.message);
      }} finally {{
        chatState.loading = false;
        document.getElementById('aiChatSendBtn').disabled = false;
        input.focus();
      }}
    }}

    function openChatSettings() {{
      const s = getChatSettings();
      document.getElementById('chatSettingBaseUrl').value = s.baseUrl || '';
      document.getElementById('chatSettingModel').value = s.model || '';
      document.getElementById('chatSettingApiKey').value = s.apiKey || '';
      document.getElementById('chatSettingsOverlay').classList.add('open');
    }}

    function closeChatSettings() {{
      document.getElementById('chatSettingsOverlay').classList.remove('open');
    }}

    function saveAndCloseChatSettings() {{
      saveChatSettings({{
        baseUrl: document.getElementById('chatSettingBaseUrl').value.trim(),
        model: document.getElementById('chatSettingModel').value.trim(),
        apiKey: document.getElementById('chatSettingApiKey').value.trim()
      }});
      closeChatSettings();
    }}
  </script>

  <!-- AI Chat Overlay -->
  <div class="ai-chat-overlay" id="aiChatOverlay">
    <div class="ai-chat-panel">
      <div class="ai-chat-header">
        <h3 id="aiChatTitle">AI 对话</h3>
        <div class="chat-actions">
          <button onclick="openChatSettings()">⚙️ 配置</button>
          <button onclick="closeAiChat()">✕ 关闭</button>
        </div>
      </div>
      <div class="ai-chat-messages" id="aiChatMessages"></div>
      <div class="ai-chat-input">
        <textarea id="aiChatInput" rows="1" placeholder="输入你的问题..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){{event.preventDefault();sendChatMessage();}}"></textarea>
        <button id="aiChatSendBtn" onclick="sendChatMessage()">发送</button>
      </div>
    </div>
  </div>

  <!-- Chat Settings Overlay -->
  <div class="chat-settings-overlay" id="chatSettingsOverlay">
    <div class="chat-settings-box">
      <h3>⚙️ AI 对话模型配置</h3>
      <label>Base URL</label>
      <input type="text" id="chatSettingBaseUrl" placeholder="https://api.example.com" />
      <label>模型名称</label>
      <input type="text" id="chatSettingModel" placeholder="gpt-4o / claude-sonnet-4-20250514" />
      <label>API Key</label>
      <input type="password" id="chatSettingApiKey" placeholder="sk-..." />
      <div class="settings-actions">
        <button class="btn-cancel" onclick="closeChatSettings()">取消</button>
        <button class="btn-save" onclick="saveAndCloseChatSettings()">保存</button>
      </div>
    </div>
  </div>
</body>
</html>
"""


    # Post-process: ensure any remaining ![alt](url) in HTML body (not in <script>) are converted to <img>
    def _fix_md_images_in_body(html_text: str) -> str:
        """Replace markdown images in HTML body only, preserving them in <script> blocks."""
        parts = []
        last_end = 0
        for m in re.finditer(r'<script[^>]*>.*?</script>', html_text, re.DOTALL):
            # Fix HTML body part before this script
            body_chunk = html_text[last_end:m.start()]
            body_chunk = re.sub(
                r'!\[([^\]]*)\]\(([^)]+)\)',
                r'<img src="\2" alt="\1" style="max-width: 100%; height: auto;" />',
                body_chunk
            )
            parts.append(body_chunk)
            parts.append(m.group(0))  # Keep script block as-is
            last_end = m.end()
        # Fix remaining body after last script
        tail = html_text[last_end:]
        tail = re.sub(
            r'!\[([^\]]*)\]\(([^)]+)\)',
            r'<img src="\2" alt="\1" style="max-width: 100%; height: auto;" />',
            tail
        )
        parts.append(tail)
        return ''.join(parts)

    html = _fix_md_images_in_body(html)

    # Write HTML, fixing \r\n normalization issue
    html_bytes = html.encode('utf-8')
    # Restore CR+LF regex pattern that Python text mode strips
    html_bytes = html_bytes.replace(b'__NORMALIZED_LINE__',
        b"const normalized = raw.replace(/\r\n/g, '\n').replace(/\r/g, '\n');")
    output_path.write_bytes(html_bytes)


def build_quiz_site(
    html_content: str,
    output_dir: str,
    config: dict,
    question_type: str = 'choice',
    model_config_name: Optional[str] = None,
    custom_model: Optional[Dict] = None,
    progress_callback: Optional[Callable] = None,
    log_callback: Optional[Callable] = None,
) -> None:
    """
    构建刷题网站的主入口函数。
    
    参数:
        html_content: 抓取到的 HTML 页面内容
        output_dir: 输出目录路径
        config: 配置字典（从 config.json 加载）
        question_type: 题目类型 ('choice' 或 'answer')
        model_config_name: 模型配置名称（对应 config['models'] 中的 key）
        progress_callback: 进度回调函数 callback(stage, current, total, message)
    """
    from lib.fetcher import extract_next_data

    output_path = Path(output_dir)
    ensure_output_dir(output_path)

    def report(stage, current, total, message=''):
        if progress_callback:
            progress_callback(stage, current, total, message)

    def log(level, message):
        if log_callback:
            log_callback(level, message)

    report('初始化', 0, 100, '正在解析题目...')
    log('info', '开始解析 HTML 页面...')

    # 提取题目
    next_data = extract_next_data(html_content)
    meta, questions, knowledge_map = extract_questions(next_data)
    log('info', f'题目解析完成 | 题目数: {len(questions)} | 知识点数: {len(knowledge_map)}')
    if len(questions) == 0:
        log('warn', '未提取到任何题目，请检查 URL 是否正确')

    # 加载提示词模板
    project_dir = Path(__file__).resolve().parent.parent
    if question_type == 'answer':
        prompt_file = project_dir / "prompts" / "answer.md"
    else:
        prompt_file = project_dir / "prompts" / "question.md"
    question_template = load_prompt(prompt_file)

    knowledge_prompt_file = project_dir / "prompts" / "knowledge.md"
    if knowledge_prompt_file.is_file():
        knowledge_template = load_prompt(knowledge_prompt_file)
    else:
        knowledge_template = DEFAULT_KNOWLEDGE_PROMPT

    # 初始化 LLM 客户端
    llm_config = config.get("llm", {})

    if custom_model:
        # 用户自定义模型（前端直接传入）
        api_key = custom_model.get("api_key") or llm_config.get("api_key") or os.getenv("YOURAPI_API_KEY")
        if not api_key:
            raise RuntimeError("请提供 API Key")
        client = LLMClient(
            api_key=api_key,
            model=custom_model.get("model", "gpt-4o"),
            temperature=llm_config.get("temperature", 0.2),
            max_retries=llm_config.get("max_retries", 3),
            retry_wait=llm_config.get("retry_wait", 1.0),
            timeout=llm_config.get("timeout", 60.0),
            base_url=custom_model.get("base_url"),
            log_callback=log,
        )
    elif model_config_name:
        models_config = config.get("models", {})
        selected_model = models_config.get(model_config_name)
        if not selected_model:
            raise RuntimeError(f"模型配置 '{model_config_name}' 不存在，可用: {list(models_config.keys())}")
        api_key = selected_model.get("api_key") or llm_config.get("api_key") or os.getenv("YOURAPI_API_KEY")
        if not api_key:
            raise RuntimeError("请提供 API Key")
        client = LLMClient(
            api_key=api_key,
            model=selected_model.get("model", "gemini-3-pro"),
            temperature=llm_config.get("temperature", 0.2),
            max_retries=llm_config.get("max_retries", 3),
            retry_wait=llm_config.get("retry_wait", 1.0),
            timeout=llm_config.get("timeout", 60.0),
            base_url=selected_model.get("base_url"),
            log_callback=log,
        )
    else:
        api_key = llm_config.get("api_key") or os.getenv("YOURAPI_API_KEY")
        if not api_key:
            raise RuntimeError("请提供 API Key")
        client = LLMClient(
            api_key=api_key,
            model=llm_config.get("model", "gemini-3-pro"),
            temperature=llm_config.get("temperature", 0.2),
            max_retries=llm_config.get("max_retries", 3),
            retry_wait=llm_config.get("retry_wait", 1.0),
            timeout=llm_config.get("timeout", 60.0),
            base_url=llm_config.get("base_url"),
            log_callback=log,
        )

    concurrency = llm_config.get("concurrency", 3)
    log('info', f'LLM 客户端初始化 | model={client.model} | base_url={client.base_url} | concurrency={concurrency}')

    # 加载缓存
    question_responses_path = output_path / "llm_question_responses.json"
    knowledge_responses_path = output_path / "llm_knowledge_responses.json"
    question_prompts_path = output_path / "question_prompts.json"

    question_responses = load_json(question_responses_path)
    knowledge_responses = load_json(knowledge_responses_path)
    question_prompts = load_json(question_prompts_path)

    # 处理题目
    _save_lock = threading.Lock()
    _save_counter = [0]
    _SAVE_EVERY = 5

    report('生成题目解析', 0, len(questions), f'AI 解析题目中... 0/{len(questions)}')
    log('info', f'开始生成题目解析 | 共 {len(questions)} 题 | 并发 {concurrency}')

    def process_question(question):
        key = question_key(question)
        if key in question_responses and question_responses[key]:
            return key, question_responses[key], False
        prompt = build_question_prompt(question_template, meta, question)
        question_prompts[key] = prompt
        try:
            response = client.chat([{"role": "user", "content": prompt}])
            return key, response, True
        except Exception as e:
            log('error', f'题目 {key} 生成失败: {e}')
            return key, "", True

    q_done = [0]
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(process_question, q) for q in questions]
        for future in as_completed(futures):
            key, response, is_new = future.result()
            question_responses[key] = response
            q_done[0] += 1
            report('生成题目解析', q_done[0], len(questions),
                   f'AI 解析题目中... {q_done[0]}/{len(questions)}')
            if is_new:
                with _save_lock:
                    _save_counter[0] += 1
                    if _save_counter[0] % _SAVE_EVERY == 0:
                        save_json(question_responses_path, question_responses)
                        save_json(question_prompts_path, question_prompts)

    save_json(question_responses_path, question_responses)
    save_json(question_prompts_path, question_prompts)
    log('info', f'题目解析完成 | 成功 {sum(1 for v in question_responses.values() if v)} 题')

    # 处理知识点
    report('生成知识点总结', 0, len(knowledge_map),
           f'AI 总结知识点中... 0/{len(knowledge_map)}')
    log('info', f'开始生成知识点总结 | 共 {len(knowledge_map)} 个知识点')

    def process_knowledge(kp_name, kp_questions):
        if kp_name in knowledge_responses and knowledge_responses[kp_name]:
            return kp_name, knowledge_responses[kp_name], False
        prompt = build_knowledge_prompt(knowledge_template, kp_name, meta, kp_questions, question_responses)
        try:
            response = client.chat([{"role": "user", "content": prompt}])
            return kp_name, response, True
        except Exception as e:
            log('error', f'知识点 "{kp_name}" 生成失败: {e}')
            return kp_name, "", True

    kp_done = [0]
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(process_knowledge, kp, qs) for kp, qs in knowledge_map.items()]
        for future in as_completed(futures):
            kp_name, response, is_new = future.result()
            knowledge_responses[kp_name] = response
            kp_done[0] += 1
            report('生成知识点总结', kp_done[0], len(knowledge_map),
                   f'AI 总结知识点中... {kp_done[0]}/{len(knowledge_map)}')
            if is_new:
                with _save_lock:
                    save_json(knowledge_responses_path, knowledge_responses)

    save_json(question_responses_path, question_responses)
    save_json(knowledge_responses_path, knowledge_responses)
    save_json(question_prompts_path, question_prompts)
    log('info', f'知识点总结完成 | 成功 {sum(1 for v in knowledge_responses.values() if v)} 个')

    # 构建网站数据
    for question in questions:
        key = question_key(question)
        model_response = question_responses.get(key, "")
        if isinstance(model_response, dict):
            model_response = model_response.get('response', '')
        question["model_response"] = model_response

    knowledge_points = []
    for kp_name, kp_questions in knowledge_map.items():
        summary_markdown = knowledge_responses.get(kp_name, "")
        if isinstance(summary_markdown, dict):
            summary_markdown = summary_markdown.get('response', '')
        knowledge_points.append({
            "name": kp_name,
            "related_questions": [question_key(q) for q in kp_questions],
            "summary_markdown": summary_markdown,
        })

    site_data = {
        "meta": meta,
        "questions": questions,
        "knowledge_points": knowledge_points,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    data_path = output_path / "quiz_data.json"
    html_path = output_path / "index.html"

    save_json(data_path, site_data)
    generate_html(site_data, html_path)
    log('info', f'网站构建完成 | 题目数: {len(questions)} | 知识点: {len(knowledge_points)}')

    report('完成', 1, 1, '生成完成！')
