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

        def format_inline(text: str) -> str:
            escaped = html_escape(text)
            escaped = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1" style="max-width: 100%; height: auto;" />', escaped)
            escaped = re.sub(r'!(https?://[^\s<>"\'\[\]]+(?:\.[a-zA-Z]{2,}|/[^\s<>"\'\[\]]*)?)', r'<img src="\1" alt="图片" style="max-width: 100%; height: auto;" />', escaped)
            escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)
            escaped = re.sub(r'__(.+?)__', r'<strong>\1</strong>', escaped)
            escaped = re.sub(r'`([^`]+)`', r'<code>\1</code>', escaped)
            escaped = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', escaped)
            escaped = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'<em>\1</em>', escaped)
            return escaped

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
            <article class='card knowledge-card'>
              <div class='knowledge-header'>
                <h3>{name}</h3>
                <span class='badge'>共 {count} 题</span>
              </div>
              {summary_block}
            </article>
            """.strip()
        )

    question_cards = []
    for q in site_data.get('questions', []):
        knowledge_point = html_escape(q.get('knowledge_point') or '未分类')
        prompt = html_escape(q.get('prompt') or '')
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
  <link rel="preconnect" href="https://cdn.jsdelivr.net" />
  <script src="https://cdn.jsdelivr.net/npm/markdown-it@13/dist/markdown-it.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.6.1/mermaid.min.js"></script>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
      --bg: #f5f7fb;
      --card: #ffffff;
      --accent: #2563eb;
      --shadow: 0 12px 30px -20px rgba(15, 23, 42, 0.4);
      --border: rgba(148, 163, 184, 0.25);
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: #0f172a;
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: clamp(1.2rem, 3vw, 2rem) clamp(1rem, 4vw, 2rem);
      box-sizing: border-box;
    }}
    section {{
      margin-bottom: clamp(1.8rem, 4vw, 2.8rem);
    }}
    h1, h2 {{
      margin: 0;
      font-weight: 700;
    }}
    h1 {{
      font-size: clamp(1.4rem, 2.4vw, 2.4rem);
      color: #1e293b;
    }}
    h2 {{
      font-size: clamp(1.2rem, 1.8vw, 1.6rem);
      color: #1f2937;
      margin-bottom: 1rem;
    }}
    .page-head {{
      display: flex;
      flex-direction: column;
      gap: 1rem;
      margin-bottom: clamp(1.6rem, 3vw, 2.4rem);
    }}
    .stats-container {{
      background: var(--card);
      border-radius: 18px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      box-shadow: var(--shadow);
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 1rem;
      margin-bottom: 1rem;
    }}
    .stat-item {{
      text-align: center;
      padding: 0.8rem;
      background: rgba(37, 99, 235, 0.05);
      border-radius: 12px;
    }}
    .stat-number {{
      font-size: 1.8rem;
      font-weight: 700;
      color: var(--accent);
      margin-bottom: 0.2rem;
    }}
    .stat-label {{
      font-size: 0.85rem;
      color: #64748b;
      font-weight: 500;
    }}
    .progress-bar {{
      width: 100%;
      height: 8px;
      background: rgba(148, 163, 184, 0.2);
      border-radius: 4px;
      overflow: hidden;
    }}
    .progress-fill {{
      height: 100%;
      background: linear-gradient(90deg, var(--accent), #3b82f6);
      border-radius: 4px;
      transition: width 0.3s ease;
    }}
    .quiz-progress {{
      background: #f1f5f9;
      border: 1px solid #cbd5e1;
      border-radius: 12px;
      padding: 1.2rem;
      margin-bottom: 1.5rem;
    }}
    .progress-info {{
      text-align: center;
    }}
    .progress-info span {{
      font-weight: 600;
      color: #475569;
      margin-bottom: 8px;
      display: block;
      font-size: 1rem;
    }}
    .quiz-navigation {{
      display: flex;
      justify-content: center;
      gap: 12px;
      margin-top: 20px;
      padding-top: 20px;
      border-top: 1px solid #e2e8f0;
    }}
    .nav-btn {{
      background: var(--accent);
      color: white;
      border: none;
      padding: 10px 20px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 500;
      transition: all 0.2s ease;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .nav-btn:hover:not(:disabled) {{
      background: #2563eb;
      transform: translateY(-1px);
      box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }}
    .nav-btn:disabled {{
      background: #94a3b8;
      cursor: not-allowed;
      transform: none;
      box-shadow: none;
    }}
    .toggle-mode-btn {{
      background: #059669;
    }}
    .toggle-mode-btn:hover {{
      background: #047857;
    }}
    .quiz-mode-toggle {{
      text-align: center;
      margin-bottom: 20px;
    }}
    .meta-line {{
      color: #64748b;
      font-size: 0.95rem;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
    }}
    .controls select,
    .controls input {{
      flex: 1 1 240px;
      min-width: 160px;
      padding: 0.65rem 0.8rem;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #ffffff;
      font-size: 0.95rem;
      box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.08);
    }}
    .card {{
      background: var(--card);
      border-radius: 18px;
      padding: clamp(1rem, 3vw, 1.4rem);
      margin-bottom: 1rem;
      box-shadow: var(--shadow);
      border: 1px solid transparent;
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    .knowledge-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 1.2rem;
    }}
    .knowledge-card {{
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
      border-left: 4px solid rgba(37, 99, 235, 0.35);
      padding-left: clamp(0.9rem, 2vw, 1.2rem);
    }}
    .knowledge-card.active {{
      border-color: var(--accent);
      box-shadow: 0 18px 30px -22px rgba(37, 99, 235, 0.6);
    }}
    .knowledge-header {{
      display: flex;
      gap: 0.75rem;
      align-items: flex-start;
      justify-content: space-between;
      flex-wrap: wrap;
    }}
    .badge {{
      background: rgba(37, 99, 235, 0.12);
      color: var(--accent);
      padding: 0.25rem 0.65rem;
      border-radius: 999px;
      font-size: 0.75rem;
      white-space: nowrap;
    }}
    .knowledge-details {{
      background: rgba(148, 163, 184, 0.12);
      border-radius: 12px;
      padding: 1rem 1.2rem;
      margin-top: 0.8rem;
    }}
    .knowledge-details .markdown {{
      margin-top: 0.8rem;
      min-height: 200px;
      max-height: none;
      overflow: visible;
    }}
    .knowledge-details[open] .markdown {{
      padding: 0.5rem 0;
    }}
    .question-card {{
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }}
    .question-header {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem 1rem;
      align-items: center;
    }}
    .question-header h3 {{
      margin: 0;
      font-size: clamp(1.05rem, 2vw, 1.2rem);
      color: #1f2937;
    }}
    .question-card p {{
      margin: 0;
      color: #334155;
      line-height: 1.65;
    }}
    .sub-question {{
      margin-top: 0.4rem;
      padding-top: 0.75rem;
      border-top: 1px dashed rgba(148, 163, 184, 0.35);
    }}
    .sub-question h4 {{
      margin: 0 0 0.4rem;
      font-size: 0.95rem;
      color: #0f172a;
    }}
    ul.options {{
      list-style: none;
      padding: 0;
      margin: 0.5rem 0;
      color: #1e293b;
    }}
    ul.options li {{
      margin-bottom: 0.35rem;
    }}
    ul.interactive-options {{
      list-style: none;
      padding: 0;
      margin: 0.5rem 0;
    }}
    .option-item {{
      margin-bottom: 0.5rem;
      border-radius: 8px;
      transition: background-color 0.2s ease;
    }}
    .option-item:hover {{
      background-color: rgba(37, 99, 235, 0.05);
    }}
    .option-label {{
      display: flex;
      align-items: center;
      padding: 0.6rem 0.8rem;
      cursor: pointer;
      border-radius: 8px;
      transition: all 0.2s ease;
    }}
    .option-input {{
      margin-right: 0.8rem;
      transform: scale(1.1);
    }}
    .option-text {{
      flex: 1;
      color: #1e293b;
      line-height: 1.5;
    }}
    .option-item.selected {{
      background-color: rgba(37, 99, 235, 0.1);
      border: 1px solid rgba(37, 99, 235, 0.3);
    }}
    .option-item.correct {{
      background-color: rgba(34, 197, 94, 0.1);
      border: 1px solid rgba(34, 197, 94, 0.3);
    }}
    .option-item.incorrect {{
      background-color: rgba(239, 68, 68, 0.1);
      border: 1px solid rgba(239, 68, 68, 0.3);
    }}
    .question-actions {{
      margin: 1rem 0;
      display: flex;
      gap: 0.8rem;
      flex-wrap: wrap;
    }}
    .submit-btn, .show-answer-btn {{
      padding: 0.6rem 1.2rem;
      border: none;
      border-radius: 8px;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .submit-btn {{
      background: var(--accent);
      color: white;
    }}
    .submit-btn:hover {{
      background: #1d4ed8;
      transform: translateY(-1px);
    }}
    .submit-btn:disabled {{
      background: #94a3b8;
      cursor: not-allowed;
      transform: none;
    }}
    .show-answer-btn {{
      background: #f59e0b;
      color: white;
    }}
    .show-answer-btn:hover {{
      background: #d97706;
      transform: translateY(-1px);
    }}
    .answer-feedback {{
      padding: 0.8rem 1rem;
      border-radius: 8px;
      margin: 0.5rem 0;
      font-weight: 600;
    }}
    .answer-feedback.correct {{
      background: rgba(34, 197, 94, 0.1);
      color: #059669;
      border: 1px solid rgba(34, 197, 94, 0.3);
    }}
    .answer-feedback.incorrect {{
      background: rgba(239, 68, 68, 0.1);
      color: #dc2626;
      border: 1px solid rgba(239, 68, 68, 0.3);
    }}
    .correct-answer {{
      background: rgba(34, 197, 94, 0.1);
      padding: 0.6rem 0.8rem;
      border-radius: 8px;
      border: 1px solid rgba(34, 197, 94, 0.3);
      margin: 0.5rem 0;
    }}
    .answer-analysis {{
      margin-top: 0.8rem;
    }}
    .rich-details {{
      background: rgba(148, 163, 184, 0.12);
      border-radius: 12px;
      padding: 0.6rem 0.85rem;
    }}
    .rich-details summary {{
      cursor: pointer;
      font-weight: 600;
      color: var(--accent);
    }}
    .rich-details .markdown {{
      margin-top: 0.6rem;
    }}
    .markdown {{
      line-height: 1.68;
      color: inherit;
    }}
    .markdown table {{
      width: 100%;
      border-collapse: collapse;
      margin: 0.75rem 0;
      font-size: 0.95rem;
    }}
    .markdown th,
    .markdown td {{
      border: 1px solid rgba(148, 163, 184, 0.4);
      padding: 0.5rem 0.65rem;
    }}
    .markdown th {{
      background: rgba(37, 99, 235, 0.06);
      font-weight: 600;
    }}
    .markdown code {{
      background: rgba(148, 163, 184, 0.2);
      border-radius: 6px;
      padding: 0.15rem 0.3rem;
      font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
    }}
    .markdown pre {{
      background: rgba(15, 23, 42, 0.85);
      color: #e2e8f0;
      padding: 0.9rem;
      border-radius: 12px;
      overflow-x: auto;
      font-size: 0.9rem;
    }}
    .markdown.empty {{
      color: #94a3b8;
    }}
    .mermaid {{
      text-align: center;
      margin: 1rem 0;
      background: #ffffff;
      border-radius: 8px;
      padding: 1rem;
      border: 1px solid rgba(148, 163, 184, 0.2);
    }}
    @media (max-width: 720px) {{
      main {{
        padding: 1rem 0.9rem 1.4rem;
      }}
      .controls select,
      .controls input {{
        flex: 1 1 100%;
        min-width: 100%;
      }}
      .knowledge-grid {{
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1rem;
      }}
      .question-header {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .question-header h3 {{
        font-size: 1.05rem;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="page-head">
      <h1>{site_data['meta'].get('paper_name', '刷题笔记')}</h1>
      <p class="meta-line">题目总数：{site_data['meta'].get('item_count')} ｜ 生成时间：{site_data['generated_at']}</p>
      <div class="controls">
        <select id="knowledgeFilter"></select>
        <input id="searchInput" type="search" placeholder="搜索题干 / 解析 / AI 笔记" />
      </div>
    </section>
    
    <section class="stats-section">
      <div class="stats-container">
        <h2>答题统计</h2>
        <div id="statsContainer">
          <div class="stats-grid">
            <div class="stat-item">
              <div class="stat-number">0</div>
              <div class="stat-label">总题数</div>
            </div>
            <div class="stat-item">
              <div class="stat-number">0</div>
              <div class="stat-label">已答题</div>
            </div>
            <div class="stat-item">
              <div class="stat-number">0</div>
              <div class="stat-label">答对</div>
            </div>
            <div class="stat-item">
              <div class="stat-number">0</div>
              <div class="stat-label">答错</div>
            </div>
            <div class="stat-item">
              <div class="stat-number">0%</div>
              <div class="stat-label">正确率</div>
            </div>
            <div class="stat-item">
              <div class="stat-number">0%</div>
              <div class="stat-label">进度</div>
            </div>
          </div>
          <div class="progress-bar">
            <div class="progress-fill" style="width: 0%"></div>
          </div>
        </div>
      </div>
    </section>
    <section class="knowledge-section">
      <h2>知识点总结</h2>
      <div id="knowledgeContainer" class="knowledge-grid">{knowledge_initial}</div>
    </section>
    <section class="question-section">
      <h2>题目列表</h2>
      <div id="questionsContainer">{questions_initial}</div>
    </section>
  </main>
  <script>
    const DATA = {data_json};

    function renderBasicMarkdown(raw = '') {{
      if (!raw) return '<p>（尚未生成，稍后重试）</p>';
      const normalized = raw.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
      const lines = normalized.split('\n');
      const htmlParts = [];
      let inUl = false;
      let inOl = false;
      const closeLists = () => {{
        if (inUl) {{ htmlParts.push('</ul>'); inUl = false; }}
        if (inOl) {{ htmlParts.push('</ol>'); inOl = false; }}
      }};
      const formatInline = (text) => {{
        const escaped = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        if (text.includes('!http')) {{
          console.log('Processing image text:', text);
        }}
        const result = escaped
          .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" style="max-width: 100%; height: auto;" />')
          .replace(/!(https?:\/\/[^\s<>"\'\\[\]]+(?:\.[a-zA-Z]{{2,}}|\/[^\s<>"\'\\[\]]*)?)/g, '<img src="$1" alt="图片" style="max-width: 100%; height: auto;" />');
        if (text.includes('!http') && result !== escaped) {{
          console.log('Image conversion result:', result);
        }}
        return result
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
      stats: {{
        total: 0,
        answered: 0,
        correct: 0,
        incorrect: 0
      }},
      answeredQuestions: {{}}
    }};

    function saveState() {{
      try {{
        const toSave = {{
          currentQuestionIndex: state.currentQuestionIndex,
          quizMode: state.quizMode,
          knowledge: state.knowledge,
          answeredQuestions: state.answeredQuestions,
          scrollY: window.scrollY
        }};
        localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave));
      }} catch(e) {{}}
    }}

    function loadState() {{
      try {{
        const saved = localStorage.getItem(STORAGE_KEY);
        if (!saved) return;
        const parsed = JSON.parse(saved);
        if (typeof parsed.currentQuestionIndex === 'number') state.currentQuestionIndex = parsed.currentQuestionIndex;
        if (typeof parsed.quizMode === 'boolean') state.quizMode = parsed.quizMode;
        if (parsed.knowledge) state.knowledge = parsed.knowledge;
        if (parsed.answeredQuestions) state.answeredQuestions = parsed.answeredQuestions;
        if (typeof parsed.scrollY === 'number') state._savedScrollY = parsed.scrollY;
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

    function renderKnowledge() {{
      const container = document.getElementById('knowledgeContainer');
      container.innerHTML = '';
      DATA.knowledge_points.forEach(({{name, summary_markdown, related_questions}}) => {{
        const card = document.createElement('article');
        card.className = 'card knowledge-card' + (state.knowledge === name ? ' active' : '');
        const totalText = `<span class="badge">共 ${{related_questions.length}} 题</span>`;
        const summaryHTML = summary_markdown && typeof summary_markdown === 'string' ? renderMarkdownWithMermaid(summary_markdown) : '<p class="empty-text">（尚未生成，稍后重试）</p>';
        const summaryBlock = summary_markdown
          ? `<details class="knowledge-details"><summary>查看大模型总结</summary><div class="markdown">${{summaryHTML}}</div></details>`
          : `<div class="markdown empty">${{summaryHTML}}</div>`;
        card.innerHTML = `
          <div class='knowledge-header'>
            <h3>${{name}}</h3>
            ${{totalText}}
          </div>
          ${{summaryBlock}}
        `;
        container.append(card);
      }});
    }}

    function filterQuestions() {{
      return DATA.questions.filter((q) => {{
        const matchKnowledge = state.knowledge === 'all' || q.knowledge_point === state.knowledge;
        if (!matchKnowledge) return false;
        if (!state.search) return true;
        const haystack = [
          q.prompt,
          q.knowledge_point,
          q.model_response || '',
          ...(q.sub_questions || []).map((sub) => [
            sub.official_analysis || '',
            (sub.options || []).map((opt) => opt.text).join(' ')
          ].join(' '))
        ].join(' ').toLowerCase();
        return haystack.includes(state.search.toLowerCase());
      }});
    }}

    function renderQuestions() {{
      const container = document.getElementById('questionsContainer');
      container.innerHTML = '';
      const results = filterQuestions();
      if (!results.length) {{
        container.innerHTML = '<p>未找到匹配的题目。</p>';
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
    }}

    function renderSingleQuestion(questions) {{
      const container = document.getElementById('questionsContainer');
      const currentQuestion = questions[state.currentQuestionIndex];
      if (!currentQuestion) {{
        container.innerHTML = '<p>没有更多题目了。</p>';
        return;
      }}
      const card = document.createElement('article');
      card.className = 'card question-card';
      const difficultyBadge = currentQuestion.difficulty ? `<span class="badge">难度：${{currentQuestion.difficulty}}</span>` : '';
      const progressIndicator = `
        <div class="quiz-progress">
          <div class="progress-info">
            <span>第 ${{state.currentQuestionIndex + 1}} 题 / 共 ${{questions.length}} 题</span>
            <div class="progress-bar">
              <div class="progress-fill" style="width: ${{((state.currentQuestionIndex + 1) / questions.length * 100).toFixed(1)}}%"></div>
            </div>
          </div>
        </div>
      `;
      const header = `
        ${{progressIndicator}}
        <div class='question-header'>
          <h3>第${{currentQuestion.index}}题（题号：${{currentQuestion.sort || currentQuestion.index}}）</h3>
          <span class='badge'>${{currentQuestion.knowledge_point}}</span>
          ${{difficultyBadge}}
        </div>
        <p>${{currentQuestion.prompt}}</p>
      `;
      const subs = (currentQuestion.sub_questions || []).map((sub, subIdx) => {{
        const correctCount = (sub.options || []).filter(opt => opt.is_correct).length;
        const isMultiple = correctCount > 1;
        const inputType = isMultiple ? 'checkbox' : 'radio';
        const questionId = `q${{currentQuestion.id || currentQuestion.index}}_${{subIdx}}`;
        const options = (sub.options || []).map((opt) => {{
          const optionId = `${{questionId}}_${{opt.label}}`;
          return `
            <li class="option-item">
              <label for="${{optionId}}" class="option-label">
                <input type="${{inputType}}" id="${{optionId}}" name="${{questionId}}" value="${{opt.label}}" class="option-input" />
                <span class="option-text">${{opt.label}}. ${{opt.text}}</span>
              </label>
            </li>
          `;
        }}).join('');
        const answers = sub.correct_letters && sub.correct_letters.length
          ? `<p class="correct-answer" style="display: none;"><strong>参考答案：</strong>${{sub.correct_letters.join('、')}}</p>`
          : '';
        const official = sub.official_analysis
          ? `<details class="rich-details answer-analysis" style="display: none;"><summary>官方解析</summary><div class="markdown">${{md.render(sub.official_analysis)}}</div></details>`
          : '';
        const subQuestionText = sub.question
          ? `<div class="sub-question-text">${{md.render(sub.question)}}</div>`
          : '';
        const correctAnswerDetail = sub.correct_answer
          ? `<details class="rich-details answer-analysis" style="display: none;"><summary>参考答案</summary><div class="markdown">${{md.render(sub.correct_answer)}}</div></details>`
          : '';
        const hasOptions = sub.options && sub.options.length > 0;
        const interactionHtml = hasOptions
          ? `<ul class='options interactive-options'>${{options}}</ul>
             <div class="question-actions">
               <button class="submit-btn" onclick="submitAnswer('${{questionId}}')">提交答案</button>
               <button class="show-answer-btn" onclick="showAnswer('${{questionId}}')" style="display: none;">查看答案</button>
             </div>`
          : `<div class="question-actions">
               <button class="show-answer-btn" onclick="showAnswer('${{questionId}}')" style="display: inline-block;">查看答案</button>
             </div>`;
        return `
          <div class='sub-question' data-question-id="${{questionId}}" data-correct-answers="${{(sub.correct_letters || []).join(',')}}" data-is-multiple="${{isMultiple}}">
            <h4>子题${{sub.label}}</h4>
            ${{subQuestionText}}
            ${{interactionHtml}}
            <div class="answer-feedback" style="display: none;"></div>
            ${{answers}}
            ${{correctAnswerDetail}}
            ${{official}}
          </div>
        `;
      }}).join('');
      const navigation = `
        <div class="quiz-navigation">
          <button class="nav-btn" onclick="previousQuestion()" ${{state.currentQuestionIndex === 0 ? 'disabled' : ''}}>上一题</button>
          <button class="nav-btn" onclick="nextQuestion()" ${{state.currentQuestionIndex >= questions.length - 1 ? 'disabled' : ''}}>下一题</button>
          <button class="nav-btn toggle-mode-btn" onclick="toggleQuizMode()">显示所有题目</button>
        </div>
      `;
      const aiNote = currentQuestion.model_response
        ? `<details class="rich-details"><summary>AI 记忆笔记</summary><div class="markdown">${{md.render(currentQuestion.model_response)}}</div></details>`
        : '<details class="rich-details"><summary>AI 记忆笔记</summary><div class="markdown empty">暂无内容</div></details>';
      card.innerHTML = header + subs + navigation + aiNote;
      container.append(card);
    }}

    function renderAllQuestions(results) {{
      const container = document.getElementById('questionsContainer');
      const toggleButton = document.createElement('div');
      toggleButton.className = 'quiz-mode-toggle';
      toggleButton.innerHTML = '<button class="nav-btn toggle-mode-btn" onclick="toggleQuizMode()">单题模式</button>';
      container.appendChild(toggleButton);
      results.forEach((q) => {{
        const card = document.createElement('article');
        card.className = 'card question-card';
        const difficultyBadge = q.difficulty ? `<span class="badge">难度：${{q.difficulty}}</span>` : '';
        const header = `
          <div class='question-header'>
            <h3>第${{q.index}}题（题号：${{q.sort || q.index}}）</h3>
            <span class='badge'>${{q.knowledge_point}}</span>
            ${{difficultyBadge}}
          </div>
          <p>${{q.prompt}}</p>
        `;
        const subs = (q.sub_questions || []).map((sub, subIdx) => {{
          const correctCount = (sub.options || []).filter(opt => opt.is_correct).length;
          const isMultiple = correctCount > 1;
          const inputType = isMultiple ? 'checkbox' : 'radio';
          const questionId = `q${{q.id || q.index}}_${{subIdx}}`;
          const options = (sub.options || []).map((opt) => {{
            const optionId = `${{questionId}}_${{opt.label}}`;
            return `
              <li class="option-item">
                <label for="${{optionId}}" class="option-label">
                  <input type="${{inputType}}" id="${{optionId}}" name="${{questionId}}" value="${{opt.label}}" class="option-input" />
                  <span class="option-text">${{opt.label}}. ${{opt.text}}</span>
                </label>
              </li>
            `;
          }}).join('');
          const answers = sub.correct_letters && sub.correct_letters.length
            ? `<p class="correct-answer" style="display: none;"><strong>参考答案：</strong>${{sub.correct_letters.join('、')}}</p>`
            : '';
          const official = sub.official_analysis
            ? `<details class="rich-details answer-analysis" style="display: none;"><summary>官方解析</summary><div class="markdown">${{md.render(sub.official_analysis)}}</div></details>`
            : '';
          const subQuestionText = sub.question
            ? `<div class="sub-question-text">${{md.render(sub.question)}}</div>`
            : '';
          const correctAnswerDetail = sub.correct_answer
            ? `<details class="rich-details answer-analysis" style="display: none;"><summary>参考答案</summary><div class="markdown">${{md.render(sub.correct_answer)}}</div></details>`
            : '';
          const hasOptions = sub.options && sub.options.length > 0;
          const interactionHtml = hasOptions
            ? `<ul class='options interactive-options'>${{options}}</ul>
               <div class="question-actions">
                 <button class="submit-btn" onclick="submitAnswer('${{questionId}}')">提交答案</button>
                 <button class="show-answer-btn" onclick="showAnswer('${{questionId}}')" style="display: none;">查看答案</button>
               </div>`
            : `<div class="question-actions">
                 <button class="show-answer-btn" onclick="showAnswer('${{questionId}}')" style="display: inline-block;">查看答案</button>
               </div>`;
          return `
            <div class='sub-question' data-question-id="${{questionId}}" data-correct-answers="${{(sub.correct_letters || []).join(',')}}" data-is-multiple="${{isMultiple}}">
              <h4>子题${{sub.label}}</h4>
              ${{subQuestionText}}
              ${{interactionHtml}}
              <div class="answer-feedback" style="display: none;"></div>
              ${{answers}}
              ${{correctAnswerDetail}}
              ${{official}}
            </div>
          `;
        }}).join('');
        const aiNote = q.model_response
          ? `<details class="rich-details"><summary>AI 记忆笔记</summary><div class="markdown">${{md.render(q.model_response)}}</div></details>`
          : '<details class="rich-details"><summary>AI 记忆笔记</summary><div class="markdown empty">暂无内容</div></details>';
        card.innerHTML = header + subs + aiNote;
        container.append(card);
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
      select.addEventListener('change', (event) => {{
        state.knowledge = event.target.value;
        state.currentQuestionIndex = 0;
        saveState();
        renderKnowledge();
        renderQuestions();
      }});
    }}

    function registerSearch() {{
      const input = document.getElementById('searchInput');
      input.addEventListener('input', (event) => {{
        state.search = event.target.value.trim();
        renderQuestions();
      }});
    }}

    function initMermaid() {{
      if (window.mermaid) {{
        mermaid.initialize({{
          startOnLoad: false,
          theme: 'default',
          themeVariables: {{
            primaryColor: '#2563eb',
            primaryTextColor: '#1f2937',
            primaryBorderColor: '#3b82f6',
            lineColor: '#6b7280',
            secondaryColor: '#f3f4f6',
            tertiaryColor: '#ffffff'
          }}
        }});
      }}
    }}

    function renderMermaidCharts() {{
      if (window.mermaid) {{
        const mermaidElements = document.querySelectorAll('.mermaid');
        mermaidElements.forEach((element, index) => {{
          const graphDefinition = element.textContent;
          const id = `mermaid-${{Date.now()}}-${{index}}`;
          element.id = id;
          mermaid.render(id + '-svg', graphDefinition).then((result) => {{
            element.innerHTML = result.svg;
          }}).catch((error) => {{
            console.error('Mermaid rendering error:', error);
            element.innerHTML = '<p style="color: #ef4444;">图表渲染失败</p>';
          }});
        }});
      }}
    }}

    function renderMarkdownWithMermaid(content) {{
      if (!content) return '<p>（尚未生成，稍后重试）</p>';
      const mermaidRegex = /```mermaid\n([\s\S]*?)\n```/g;
      let processedContent = content.replace(mermaidRegex, (match, graphDef) => {{
        const cleanGraphDef = graphDef.trim();
        return `<div class="mermaid">${{cleanGraphDef}}</div>`;
      }});
      const htmlContent = md.render ? md.render(processedContent) : renderBasicMarkdown(processedContent);
      return htmlContent;
    }}

    function updateStats(questions) {{
      const totalSubQuestions = questions.reduce((sum, q) => sum + (q.sub_questions || []).length, 0);
      const answered = Object.keys(state.answeredQuestions).length;
      const correct = Object.values(state.answeredQuestions).filter(a => a.isCorrect).length;
      const incorrect = answered - correct;
      state.stats = {{
        total: totalSubQuestions,
        answered: answered,
        correct: correct,
        incorrect: incorrect
      }};
      renderStatsDisplay();
    }}

    function renderStatsDisplay() {{
      const statsContainer = document.getElementById('statsContainer');
      if (!statsContainer) return;
      const {{ total, answered, correct, incorrect }} = state.stats;
      const accuracy = answered > 0 ? Math.round((correct / answered) * 100) : 0;
      const progress = total > 0 ? Math.round((answered / total) * 100) : 0;
      statsContainer.innerHTML = `
        <div class="stats-grid">
          <div class="stat-item">
            <div class="stat-number">${{total}}</div>
            <div class="stat-label">总题数</div>
          </div>
          <div class="stat-item">
            <div class="stat-number">${{answered}}</div>
            <div class="stat-label">已答题</div>
          </div>
          <div class="stat-item">
            <div class="stat-number">${{correct}}</div>
            <div class="stat-label">答对</div>
          </div>
          <div class="stat-item">
            <div class="stat-number">${{incorrect}}</div>
            <div class="stat-label">答错</div>
          </div>
          <div class="stat-item">
            <div class="stat-number">${{accuracy}}%</div>
            <div class="stat-label">正确率</div>
          </div>
          <div class="stat-item">
            <div class="stat-number">${{progress}}%</div>
            <div class="stat-label">进度</div>
          </div>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" style="width: ${{progress}}%"></div>
        </div>
      `;
    }}

    function submitAnswer(questionId) {{
      const questionDiv = document.querySelector(`[data-question-id="${{questionId}}"]`);
      if (!questionDiv) return;
      const isMultiple = questionDiv.dataset.isMultiple === 'true';
      const correctAnswers = questionDiv.dataset.correctAnswers.split(',').filter(Boolean);
      const inputs = questionDiv.querySelectorAll('input[type="radio"], input[type="checkbox"]');
      const selectedAnswers = Array.from(inputs).filter(input => input.checked).map(input => input.value);
      if (selectedAnswers.length === 0) {{
        alert('请选择答案后再提交！');
        return;
      }}
      inputs.forEach(input => input.disabled = true);
      const isCorrect = correctAnswers.length === selectedAnswers.length && 
                       correctAnswers.every(answer => selectedAnswers.includes(answer));
      const feedbackDiv = questionDiv.querySelector('.answer-feedback');
      feedbackDiv.style.display = 'block';
      feedbackDiv.className = `answer-feedback ${{isCorrect ? 'correct' : 'incorrect'}}`;
      feedbackDiv.textContent = isCorrect ? '✓ 回答正确！' : '✗ 回答错误';
      inputs.forEach(input => {{
        const optionItem = input.closest('.option-item');
        if (input.checked) {{
          optionItem.classList.add('selected');
        }}
        if (correctAnswers.includes(input.value)) {{
          optionItem.classList.add('correct');
        }} else if (input.checked) {{
          optionItem.classList.add('incorrect');
        }}
      }});
      const submitBtn = questionDiv.querySelector('.submit-btn');
      const showAnswerBtn = questionDiv.querySelector('.show-answer-btn');
      submitBtn.style.display = 'none';
      showAnswerBtn.style.display = 'inline-block';
      state.answeredQuestions[questionId] = {{ selected: selectedAnswers, isCorrect }};
      saveState();
      if (!isCorrect) {{
        showAnswer(questionId);
      }}
      const results = filterQuestions();
      updateStats(results);
    }}

    function showAnswer(questionId) {{
      const questionDiv = document.querySelector(`[data-question-id="${{questionId}}"]`);
      if (!questionDiv) return;
      const correctAnswerP = questionDiv.querySelector('.correct-answer');
      if (correctAnswerP) {{
        correctAnswerP.style.display = 'block';
      }}
      const analysisDetailsList = questionDiv.querySelectorAll('.answer-analysis');
      analysisDetailsList.forEach(details => {{
        details.style.display = 'block';
        details.open = true;
      }});
      const showAnswerBtn = questionDiv.querySelector('.show-answer-btn');
      if (showAnswerBtn) showAnswerBtn.style.display = 'none';
    }}

    function previousQuestion() {{
      if (state.currentQuestionIndex > 0) {{
        state.currentQuestionIndex--;
        saveState();
        renderQuestions();
      }}
    }}

    function nextQuestion() {{
      const questions = filterQuestions();
      if (state.currentQuestionIndex < questions.length - 1) {{
        state.currentQuestionIndex++;
        saveState();
        renderQuestions();
      }}
    }}

    function toggleQuizMode() {{
      state.quizMode = !state.quizMode;
      if (state.quizMode) {{
        state.currentQuestionIndex = 0;
      }}
      saveState();
      renderQuestions();
    }}

    document.addEventListener('DOMContentLoaded', () => {{
      initMermaid();
      loadState();
      const filterSelect = document.getElementById('knowledgeFilter');
      if (filterSelect && state.knowledge !== 'all') filterSelect.value = state.knowledge;
      populateFilter();
      registerSearch();
      renderKnowledge();
      renderQuestions();
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
      setTimeout(() => {{
        renderMermaidCharts();
      }}, 100);
    }});
  </script>
</body>
</html>
"""

    output_path.write_text(html, encoding='utf-8')


def build_quiz_site(
    html_content: str,
    output_dir: str,
    config: dict,
    question_type: str = 'choice',
    model_config_name: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
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

    report('初始化', 0, 100, '正在解析题目...')

    # 提取题目
    next_data = extract_next_data(html_content)
    meta, questions, knowledge_map = extract_questions(next_data)

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

    if model_config_name:
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
        )

    concurrency = llm_config.get("concurrency", 3)

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

    def process_question(question):
        key = question_key(question)
        if key in question_responses:
            return key, question_responses[key], False
        prompt = build_question_prompt(question_template, meta, question)
        question_prompts[key] = prompt
        try:
            response = client.chat([{"role": "user", "content": prompt}])
            return key, response, True
        except Exception as e:
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

    # 处理知识点
    report('生成知识点总结', 0, len(knowledge_map),
           f'AI 总结知识点中... 0/{len(knowledge_map)}')

    def process_knowledge(kp_name, kp_questions):
        if kp_name in knowledge_responses:
            return kp_name, knowledge_responses[kp_name], False
        prompt = build_knowledge_prompt(knowledge_template, kp_name, meta, kp_questions, question_responses)
        try:
            response = client.chat([{"role": "user", "content": prompt}])
            return kp_name, response, True
        except Exception as e:
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

    report('完成', 1, 1, '生成完成！')
