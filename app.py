#!/usr/bin/env python3
"""
题库生成网站 - Flask后端（异步任务版）
"""
import json
import os
import shutil
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

BASE_DIR = Path("/root/timu")
DATA_DIR = Path("/root/.openclaw/workspace/data/timu")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def run_task(task_id: str, url: str, html_path: Path, question_type: str = 'choice'):
    """后台线程：运行 builder"""
    task_dir = DATA_DIR / task_id
    fetcher_config = BASE_DIR / "config" / "cheko_fetcher_config.json"
    builder_script = BASE_DIR / "scripts" / "quiz_site_builder.py"
    
    # 根据题目类型选择提示词文件
    if question_type == 'answer':
        prompt_file = BASE_DIR / "prompts" / "answer.md"
    else:
        prompt_file = BASE_DIR / "prompts" / "question.md"

    def save_info(status, error=''):
        info = {
            'id': task_id, 'url': url,
            'created_at': datetime.now().isoformat(),
            'status': status, 'error': error
        }
        (task_dir / 'info.json').write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')

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

    build_cmd = [
        "python3", str(builder_script),
        "--html", str(html_path),
        "--output-dir", str(task_dir / "site"),
        "--config", str(fetcher_config),
        "--prompt-file", str(prompt_file),
        "--knowledge-prompt-file", str(BASE_DIR / "prompts" / "knowledge.md")
    ]

    try:
        # 使用 Popen 以便实时读取输出
        process = subprocess.Popen(
            build_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1
        )
        
        # 解析输出中的进度
        import re
        question_total = 0
        question_done = 0
        knowledge_total = 0
        knowledge_done = 0
        
        for line in process.stdout:
            # 解析题目处理进度: [######------] 题目处理 5/10
            q_match = re.search(r'题目处理\s+(\d+)/(\d+)', line)
            if q_match:
                question_done = int(q_match.group(1))
                question_total = int(q_match.group(2))
                save_progress('生成题目解析', question_done, question_total, 
                            f'AI 解析题目中... {question_done}/{question_total}')
                continue
                
            # 解析知识点处理进度
            kp_match = re.search(r'知识点处理\s+(\d+)/(\d+)', line)
            if kp_match:
                knowledge_done = int(kp_match.group(1))
                knowledge_total = int(kp_match.group(2))
                save_progress('生成知识点总结', knowledge_done, knowledge_total,
                            f'AI 总结知识点中... {knowledge_done}/{knowledge_total}')
                continue
                
            # 检测是否开始新阶段
            if '开始处理' in line and '知识点' in line:
                save_progress('分析知识点', 0, 1, '正在分析知识点...')
        
        process.wait(timeout=1800)
        
        if process.returncode == 0:
            save_info('completed')
            save_progress('完成', 1, 1, '生成完成！')
        else:
            save_info('failed', process.stderr.read()[-500:] if process.stderr else '')
            save_progress('失败', 0, 1, '生成失败')
            
    except subprocess.TimeoutExpired:
        save_info('failed', '处理超时（30分钟）')
        save_progress('失败', 0, 1, '处理超时')
    except Exception as e:
        save_info('failed', str(e))
        save_progress('失败', 0, 1, str(e))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.get_json()
    url = data.get('url', '').strip()
    question_type = data.get('question_type', 'choice')  # choice or answer
    if not url:
        return jsonify({'success': False, 'error': '请输入URL'})

    task_id = str(uuid.uuid4())[:8]
    task_dir = DATA_DIR / task_id
    task_dir.mkdir(exist_ok=True)

    fetcher_config = BASE_DIR / "config" / "cheko_fetcher_config.json"
    fetcher_script = BASE_DIR / "scripts" / "cheko_fetcher.py"

    try:
        with open(fetcher_config, 'r', encoding='utf-8') as f:
            config = json.load(f)

        fetch_cmd = [
            "python3", str(fetcher_script),
            "-o", str(task_dir / "output.txt"),
            url
        ]
        cookie = config.get('cookie', '')
        if cookie:
            fetch_cmd = fetch_cmd[:2] + ["--cookie", cookie] + fetch_cmd[2:]

        result = subprocess.run(fetch_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return jsonify({'success': False, 'error': f'抓取失败: {result.stderr[:300]}'})

        html_files = list(task_dir.glob("*.html"))
        if not html_files:
            return jsonify({'success': False, 'error': '未能抓取到内容，请检查URL或Cookie'})

        # 提取页面标题
        page_title = ""
        try:
            html_content = html_files[0].read_text(encoding='utf-8')
            import re
            title_match = re.search(r'<title>([^<]+)</title>', html_content, re.IGNORECASE)
            if title_match:
                full_title = title_match.group(1).strip()
            else:
                og_match = re.search(r'property=["\']og:title["\']\s+content=["\']([^"\']+)["\']', html_content)
                if og_match:
                    full_title = og_match.group(1).strip()
                else:
                    full_title = ""
            
            # 提取中间部分：例如 "系统架构设计师综合知识题 - 学习模式模式 - 系统可靠性 | 芝士架构" -> "系统可靠性"
            if full_title:
                parts = full_title.split(' - ')
                if len(parts) >= 3:
                    page_title = parts[-2].strip()  # 取倒数第二部分
                elif len(parts) == 2:
                    page_title = parts[0].strip()
        except Exception:
            pass

        # 保存初始状态
        info = {'id': task_id, 'url': url, 'title': page_title, 'question_type': question_type, 'created_at': datetime.now().isoformat(), 'status': 'building'}
        (task_dir / 'info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')

        # 后台运行 builder
        t = threading.Thread(target=run_task, args=(task_id, url, html_files[0], question_type), daemon=True)
        t.start()

        return jsonify({'success': True, 'task_id': task_id, 'status': 'building'})

    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': '抓取超时'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/task/<task_id>')
def get_task(task_id):
    task_dir = DATA_DIR / task_id
    info_file = task_dir / 'info.json'
    if not info_file.exists():
        return jsonify({'error': '任务不存在'}), 404

    info = json.loads(info_file.read_text(encoding='utf-8'))

    # 读取进度信息
    progress_file = task_dir / 'progress.json'
    if progress_file.exists():
        info['progress'] = json.loads(progress_file.read_text(encoding='utf-8'))

    # 读取题库标题
    site_dir = task_dir / 'site'
    quiz_data_file = site_dir / 'quiz_data.json'
    if quiz_data_file.exists():
        try:
            quiz_data = json.loads(quiz_data_file.read_text(encoding='utf-8'))
            info['title'] = quiz_data.get('meta', {}).get('paper_name', '')
        except:
            pass

    # 列出生成的文件
    files = []
    if site_dir.exists():
        for f in site_dir.rglob('*'):
            if f.is_file():
                files.append(str(f.relative_to(task_dir)))
    info['files'] = files
    return jsonify(info)


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

    sample_html = BASE_DIR / "sample_input" / "cheko_673625.html"
    if not sample_html.exists():
        return jsonify({'success': False, 'error': '示例文件不存在'})

    dest = task_dir / "input.html"
    shutil.copy(sample_html, dest)

    info = {'id': task_id, 'url': 'demo', 'created_at': datetime.now().isoformat(), 'status': 'building'}
    (task_dir / 'info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')

    t = threading.Thread(target=run_task, args=(task_id, 'demo', dest), daemon=True)
    t.start()

    return jsonify({'success': True, 'task_id': task_id, 'status': 'building'})


@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory(DATA_DIR, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
