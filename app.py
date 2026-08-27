#!/usr/bin/env python3
"""
题库生成网站 - Flask后端（异步任务版）
直接调用 lib.fetcher 和 lib.builder 模块，不再使用 subprocess。
"""
import json
import os
import re
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

from lib.fetcher import fetch_and_parse, extract_next_data, build_headers_from_config, fetch_html
from lib.builder import build_quiz_site

app = Flask(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = PROJECT_DIR / "config.json"
DATA_DIR = Path("/root/.openclaw/workspace/data/timu")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    if CONFIG_FILE.is_file():
        return json.loads(CONFIG_FILE.read_text(encoding='utf-8-sig'))
    return {}


def run_task(task_id: str, url: str, html_path: Path, question_type: str = 'choice', model_config: str = None, custom_model: dict = None):
    """后台线程：运行 builder"""
    task_dir = DATA_DIR / task_id
    log_lock = threading.Lock()

    def save_info(status, error=''):
        info_file = task_dir / 'info.json'
        if info_file.exists():
            info = json.loads(info_file.read_text(encoding='utf-8'))
        else:
            info = {'id': task_id, 'url': url, 'created_at': datetime.now().isoformat()}
        info['status'] = status
        if error:
            info['error'] = error
        info_file.write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')

    def append_log(level, message):
        """追加一条日志"""
        entry = {
            'time': datetime.now().strftime('%H:%M:%S'),
            'level': level,
            'msg': message
        }
        log_file = task_dir / 'logs.jsonl'
        with log_lock:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def save_progress(stage, current, total, message=''):
        progress = {
            'stage': stage,
            'current': current,
            'total': total,
            'message': message,
            'percent': int(current * 100 / total) if total > 0 else 0
        }
        (task_dir / 'progress.json').write_text(
            json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf-8')

    save_info('building')
    save_progress('初始化', 0, 100, '正在启动...')
    append_log('info', f'任务启动 | URL: {url} | 题型: {question_type} | 模型: {custom_model.get("model", model_config or "默认") if custom_model else model_config or "默认"}')

    try:
        html_content = html_path.read_text(encoding='utf-8')
        config = load_config()
        append_log('info', f'页面抓取完成 | HTML大小: {len(html_content)} 字符')

        build_quiz_site(
            html_content=html_content,
            output_dir=str(task_dir / "site"),
            config=config,
            question_type=question_type,
            model_config_name=model_config,
            custom_model=custom_model,
            progress_callback=save_progress,
            log_callback=append_log,
        )

        save_info('completed')
        save_progress('完成', 1, 1, '生成完成！')
        append_log('info', '✅ 任务完成！')

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        save_info('failed', str(e)[-500:])
        save_progress('失败', 0, 1, str(e))
        append_log('error', f'❌ 任务失败: {e}')
        append_log('error', tb[-800:])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/models', methods=['GET'])
def get_models():
    """获取可用的模型配置列表"""
    try:
        config = load_config()
        models = config.get('models', {})
        model_list = []
        for name, model_config in models.items():
            model_list.append({
                'name': name,
                'model': model_config.get('model', ''),
                'base_url': model_config.get('base_url', '')
            })
        return jsonify({'success': True, 'models': model_list})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'models': []})


@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.get_json()
    url = data.get('url', '').strip()
    question_type = data.get('question_type', 'choice')
    model_config = data.get('model_config', None)
    custom_model = data.get('custom_model', None)  # {base_url, api_key, model}
    if not url:
        return jsonify({'success': False, 'error': '请输入URL'})

    task_id = str(uuid.uuid4())[:8]
    task_dir = DATA_DIR / task_id
    task_dir.mkdir(exist_ok=True)

    try:
        config = load_config()

        # 直接用 Python 抓取，不再 subprocess
        headers = build_headers_from_config(config)
        html_text, _ = fetch_html(url, headers)

        # 保存原始 HTML
        html_file = task_dir / "input.html"
        html_file.write_text(html_text, encoding='utf-8')

        # 提取页面标题
        page_title = ""
        try:
            next_data_match = re.search(
                r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                html_text, re.DOTALL
            )
            if next_data_match:
                next_data = json.loads(next_data_match.group(1))
                page_props = next_data.get("props", {}).get("pageProps", {})
                test_meta = page_props.get("test", {})
                source_list = test_meta.get("selects") or test_meta.get("cases") or []
                if source_list:
                    first_item = source_list[0] or {}
                    paper_name = (first_item.get("paper") or {}).get("name") or test_meta.get("paperName") or ""
                    kp_name = first_item.get("kpName") or ""
                    if paper_name and kp_name:
                        page_title = f"{paper_name} - {kp_name}"
                    elif paper_name:
                        page_title = paper_name
                    elif kp_name:
                        page_title = kp_name

            if not page_title:
                title_match = re.search(r'<title>([^<]+)</title>', html_text, re.IGNORECASE)
                if title_match:
                    full_title = title_match.group(1).strip()
                else:
                    og_match = re.search(r'property=["\']og:title["\']\s+content=["\']([^"\']+)["\']', html_text)
                    full_title = og_match.group(1).strip() if og_match else ""

                if full_title:
                    full_title = re.sub(r'\s*[|｜]\s*芝士架构$', '', full_title)
                    parts = full_title.split(' - ')
                    if len(parts) >= 3:
                        page_title = f"{parts[0].strip()} - {parts[-2].strip()}"
                    elif len(parts) == 2:
                        page_title = full_title
                    else:
                        page_title = full_title
        except Exception:
            pass

        # 保存初始状态
        info = {
            'id': task_id, 'url': url, 'title': page_title,
            'question_type': question_type,
            'model_config': custom_model.get('model', model_config) if custom_model else model_config,
            'created_at': datetime.now().isoformat(), 'status': 'building'
        }
        (task_dir / 'info.json').write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')

        # 后台运行 builder
        t = threading.Thread(
            target=run_task,
            args=(task_id, url, html_file, question_type, model_config, custom_model),
            daemon=True
        )
        t.start()

        return jsonify({'success': True, 'task_id': task_id, 'status': 'building'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/task/<task_id>')
def get_task(task_id):
    task_dir = DATA_DIR / task_id
    info_file = task_dir / 'info.json'
    if not info_file.exists():
        return jsonify({'error': '任务不存在'}), 404

    info = json.loads(info_file.read_text(encoding='utf-8'))

    progress_file = task_dir / 'progress.json'
    if progress_file.exists():
        info['progress'] = json.loads(progress_file.read_text(encoding='utf-8'))

    site_dir = task_dir / 'site'
    quiz_data_file = site_dir / 'quiz_data.json'
    if quiz_data_file.exists():
        try:
            quiz_data = json.loads(quiz_data_file.read_text(encoding='utf-8'))
            info['title'] = quiz_data.get('meta', {}).get('paper_name', '')
        except Exception:
            pass

    files = []
    if site_dir.exists():
        for f in site_dir.rglob('*'):
            if f.is_file():
                files.append(str(f.relative_to(task_dir)))
    info['files'] = files
    return jsonify(info)


@app.route('/api/task/<task_id>/logs')
def get_task_logs(task_id):
    """获取任务日志（支持 ?after=N 从第N行之后读取）"""
    task_dir = DATA_DIR / task_id
    log_file = task_dir / 'logs.jsonl'
    if not log_file.exists():
        return jsonify({'logs': [], 'total': 0})

    after = int(request.args.get('after', 0))
    lines = log_file.read_text(encoding='utf-8').strip().split('\n')
    logs = []
    for i, line in enumerate(lines):
        if i >= after and line.strip():
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                logs.append({'time': '?', 'level': 'raw', 'msg': line})
    return jsonify({'logs': logs, 'total': len(lines)})


@app.route('/api/task/<task_id>/retry', methods=['POST'])
def retry_task(task_id):
    """重试失败的任务（重新运行 builder，自动跳过已有成功结果的题目）"""
    task_dir = DATA_DIR / task_id
    if not (task_dir / 'info.json').exists():
        return jsonify({'success': False, 'error': '任务不存在'})

    info = json.loads((task_dir / 'info.json').read_text(encoding='utf-8'))

    # 找到原始 html 文件
    html_files = list(task_dir.glob('*.html'))
    if not html_files:
        return jsonify({'success': False, 'error': '原始页面文件不存在'})

    html_file = html_files[0]
    model_config = info.get('model_config')
    question_type = info.get('question_type', 'choice')

    # 解析 custom_model（如果有）
    custom_model = None
    if model_config and model_config not in ('', '默认'):
        config = load_config()
        models_cfg = config.get('models', {})
        if model_config in models_cfg:
            pass  # 预设模型
        else:
            # 可能是自定义模型名 — 从最新一次请求推断，暂用默认
            pass

    # 清空日志，重新开始
    log_file = task_dir / 'logs.jsonl'
    if log_file.exists():
        log_file.unlink()

    info['status'] = 'building'
    info.pop('error', None)
    (task_dir / 'info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')

    t = threading.Thread(
        target=run_task,
        args=(task_id, info.get('url', ''), html_file, question_type, model_config, custom_model),
        daemon=True
    )
    t.start()

    return jsonify({'success': True, 'task_id': task_id})


@app.route('/api/tasks')
def list_tasks():
    tasks = []
    for d in sorted(DATA_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        info_file = d / 'info.json'
        if info_file.exists():
            tasks.append(json.loads(info_file.read_text(encoding='utf-8')))
    return jsonify(tasks)


@app.route('/api/demo', methods=['POST'])
def demo():
    task_id = str(uuid.uuid4())[:8]
    task_dir = DATA_DIR / task_id
    task_dir.mkdir(exist_ok=True)

    # 查找 data 目录中已有的 HTML 文件作为示例
    sample_html = None
    for f in DATA_DIR.glob("*.html"):
        sample_html = f
        break

    if not sample_html or not sample_html.exists():
        return jsonify({'success': False, 'error': '示例文件不存在'})

    dest = task_dir / "input.html"
    shutil.copy(sample_html, dest)

    info = {
        'id': task_id, 'url': 'demo',
        'created_at': datetime.now().isoformat(), 'status': 'building'
    }
    (task_dir / 'info.json').write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')

    t = threading.Thread(
        target=run_task, args=(task_id, 'demo', dest, 'choice', None), daemon=True
    )
    t.start()

    return jsonify({'success': True, 'task_id': task_id, 'status': 'building'})


@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory(DATA_DIR, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
