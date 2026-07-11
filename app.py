# -*- coding: utf-8 -*-
# IGENT Backend - Flask Proxy for DeepSeek API
# Created by: T.me/sii_3 | Mr Dark
# 2023 - 2026

from flask import Flask, request, Response, send_from_directory
import os, requests, json, uuid, time

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════
#  Personality Prompt (T)
# ═══════════════════════════════════════════════════
T = r"""أنت IGENT، مساعد ذكاء اصطناعي متقدم ومتطور. 
- أنت تتحدث اللغة العربية الفصحى بشكل رئيسي، وتجيب بشكل واضح ومفصل.
- أنت مساعد مفيد، محترف، وودود في آن واحد.
- عندما يُطلب منك التفكير العميق، تُظهر سلسلة أفكارك بوضوح قبل الإجابة النهائية.
- تقدم إجابات دقيقة ومبنية على معلومات موثوقة.
- تتعامل مع الملفات المرفوعة بذكاء وتحلل محتواها إن أمكن.
- تتجنب الإجابات المختصرة إلا عند الطلب، وتفضل التفصيل المفيد.
- أسلوبك رسمي لكن غير جاف، وتحاول أن تكون مفيداً قدر الإمكان."""

# ═══════════════════════════════════════════════════
#  API Configuration (from igent.py)
# ═══════════════════════════════════════════════════
API_URL = "https://api-preview.chatgot.io/api/v1/char-gpt/conversations"
MODEL_ID = 2
INCLUDE_REASONING = True

def get_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "sec-ch-ua-platform": '"Windows"',
        "sec-ch-ua": '"Chromium";v="122", "Google Chrome";v="122", "Not(A:Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "origin": "https://deepseekfree.ai",
        "referer": "https://deepseekfree.ai/",
        "accept-language": "ar-EG,ar;q=0.9,en-US;q=0.8,en;q=0.7"
    }

# ═══════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/chat')
def chat_page():
    return send_from_directory(BASE_DIR, 'chat.html')

@app.route('/api/chat')
def chat_api():
    user_message = request.args.get('msg', '')
    if not user_message:
        return Response(
            "data: " + json.dumps({"type": "error", "text": "Empty message"}, ensure_ascii=False) + "\n\n",
            mimetype='text/event-stream'
        )

    messages = [
        {"role": "user", "content": T},
        {"role": "user", "content": user_message}
    ]

    payload = {
        "device_id": uuid.uuid4().hex,
        "model_id": MODEL_ID,
        "include_reasoning": INCLUDE_REASONING,
        "messages": messages
    }

    def generate():
        resp = None
        try:
            resp = requests.post(API_URL, json=payload, headers=get_headers(), stream=True, timeout=(15, 30))
            if resp.status_code == 200:
                last_time = time.time()
                got_data = False
                for line in resp.iter_lines(decode_unicode=True):
                    if line:
                        last_time = time.time()
                    if not line or not line.startswith('data: '):
                        if got_data and time.time() - last_time > 1.5:
                            break
                        continue
                    chunk = line[6:].strip()
                    if chunk == '[DONE]':
                        break
                    try:
                        obj = json.loads(chunk).get('data', {})
                        if isinstance(obj, dict):
                            reasoning = obj.get('reasoning_content', '')
                            content = obj.get('content', '')
                            if reasoning:
                                yield f"data: {json.dumps({'type': 'reasoning', 'text': reasoning}, ensure_ascii=False)}\n\n"
                                got_data = True
                            if content:
                                yield f"data: {json.dumps({'type': 'content', 'text': content}, ensure_ascii=False)}\n\n"
                                got_data = True
                    except:
                        pass
                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'text': f'Status {resp.status_code}'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            if resp is not None:
                resp.close()

    return Response(generate(), mimetype='text/event-stream')

# ═══════════════════════════════════════════════════
#  Run
# ═══════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
