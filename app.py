import os
import logging
import secrets
import base64
import json
import requests
import re
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response, stream_with_context
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from flask_session import Session
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ==================== Config ====================
class Config:
    SECRET_KEY = os.getenv('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable is required!")

    ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'admin123')
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
    OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'openrouter/auto')

    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/db')
    # Render provides postgres:// but SQLAlchemy 2.0 requires postgresql://
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_POOL_SIZE = 20
    SQLALCHEMY_MAX_OVERFLOW = 40
    SQLALCHEMY_POOL_PRE_PING = True

    SESSION_TYPE = 'sqlalchemy'
    SESSION_SQLALCHEMY_TABLE = 'flask_sessions'
    SESSION_PERMANENT = True
    SESSION_USE_SIGNER = True
    SESSION_KEY_PREFIX = 'ufoq_session:'
    PERMANENT_SESSION_LIFETIME = 2592000

    CACHE_TYPE = 'SimpleCache'
    CACHE_DEFAULT_TIMEOUT = 300
    RATELIMIT_ENABLED = True
    RATELIMIT_STORAGE_URI = 'memory://'
    RATELIMIT_STRATEGY = 'fixed-window'

# ==================== Models ====================
db = SQLAlchemy()

class Category(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    display_name = db.Column(db.String(100), nullable=False)
    icon = db.Column(db.String(50), default='bi-tag')
    sort_order = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PromptLibrary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(50), nullable=False, default='general')
    image_url = db.Column(db.String(500), nullable=False)
    prompt_text = db.Column(db.Text, nullable=False)
    publisher = db.Column(db.String(80), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class LibraryAd(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    text = db.Column(db.Text, nullable=False)
    image_url = db.Column(db.String(500), nullable=True)
    button_text = db.Column(db.String(100), nullable=False)
    button_link = db.Column(db.String(500), nullable=False)
    duration_seconds = db.Column(db.Integer, default=5)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SiteSetting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    status = db.Column(db.String(10), default='on')
    offline_message = db.Column(db.Text, default='الموقع تحت الصيانة حالياً.')

class UploadContribution(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(50), nullable=False, default='general')
    image_url = db.Column(db.String(500), nullable=True)
    prompt_text = db.Column(db.Text, nullable=False)
    publisher_name = db.Column(db.String(80), nullable=True)
    status = db.Column(db.String(20), default='pending')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ==================== NEW: AI Chat Models ====================
class AIModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    model_id = db.Column(db.String(200), nullable=False, unique=True)  # e.g. "openai/gpt-4o"
    provider = db.Column(db.String(50), nullable=False)  # e.g. "openai", "deepseek"
    description = db.Column(db.Text, nullable=False)
    icon = db.Column(db.String(50), default='bi-cpu')
    color = db.Column(db.String(20), default='rgba(14,165,233,0.08)')
    text_color = db.Column(db.String(20), default='#0ea5e9')
    is_active = db.Column(db.Boolean, default=True)
    is_rewriter = db.Column(db.Boolean, default=False)  # grok-4.3 style
    sort_order = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ChatThread(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(128), nullable=False, index=True)
    title = db.Column(db.String(200), nullable=True)
    model_id = db.Column(db.Integer, db.ForeignKey('ai_model.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    model = db.relationship('AIModel', backref='threads')

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    thread_id = db.Column(db.Integer, db.ForeignKey('chat_thread.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    thread = db.relationship('ChatThread', backref=db.lazyload('messages'))

# ==================== App Factory ====================
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))
app.config.from_object(Config)

db.init_app(app)
migrate = Migrate(app, db)
cache = Cache(app)
limiter = Limiter(get_remote_address, app=app, default_limits=[], storage_uri=app.config['RATELIMIT_STORAGE_URI'])
Talisman(app, force_https=False, content_security_policy={
    'default-src': ["'self'"],
    'style-src': ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com", "https://cdn.jsdelivr.net"],
    'script-src': ["'self'", "'unsafe-inline'", "https://cdn.jsdelivr.net"],
    'font-src': ["'self'", "https://fonts.gstatic.com", "https://cdn.jsdelivr.net"],
    'img-src': ["'self'", "data:", "https:"],
    'connect-src': ["'self'"]
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Helper functions ----------
def generate_csrf_token():
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_urlsafe(32)
    return session['csrf_token']

def validate_csrf_token(token):
    return token == session.get('csrf_token')

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('admin_panel'))
        return f(*args, **kwargs)
    return decorated

def get_session_id():
    """Get or create a unique session ID for chat tracking"""
    if 'chat_session_id' not in session:
        session['chat_session_id'] = secrets.token_urlsafe(32)
    return session['chat_session_id']

# ---------- Database initialization ----------
_db_initialized = False
@app.before_request
def ensure_db_initialized():
    global _db_initialized
    if not _db_initialized:
        try:
            db.create_all()
            if not SiteSetting.query.first():
                db.session.add(SiteSetting())
                db.session.commit()
            if not Category.query.first():
                defaults = [
                    Category(name='images', display_name='توليد صور', sort_order=1),
                    Category(name='writing', display_name='كتابة محتوى', sort_order=2),
                    Category(name='coding', display_name='برمجة', sort_order=3),
                    Category(name='design', display_name='تصميم UI', sort_order=4),
                    Category(name='analysis', display_name='تحليل بيانات', sort_order=5),
                    Category(name='creative', display_name='إبداعي', sort_order=6),
                ]
                for cat in defaults:
                    db.session.add(cat)
                db.session.commit()
            # Initialize default AI models
            if not AIModel.query.first():
                default_models = [
                    AIModel(name='GPT-4o', model_id='openai/gpt-4o', provider='openai',
                           description='نموذج متعدد الوسائط فائق القوة، يتفوق في الفهم العميق والإجابات الدقيقة على الأسئلة المعقدة',
                           icon='bi-robot', color='rgba(16,185,129,0.08)', text_color='#10b981', sort_order=1),
                    AIModel(name='Claude 3.5 Sonnet', model_id='anthropic/claude-3.5-sonnet', provider='anthropic',
                           description='نموذج ذكي بشكل استثنائي في الكتابة الإبداعية والتحليل العميق والبرمجة المعقدة',
                           icon='bi-stars', color='rgba(245,158,11,0.08)', text_color='#f59e0b', sort_order=2),
                    AIModel(name='DeepSeek V3', model_id='deepseek/deepseek-chat', provider='deepseek',
                           description='نموذج صيني متقدم يتفوق في البرمجة والرياضيات والاستدلال المنطقي العميق',
                           icon='bi-code-slash', color='rgba(99,102,241,0.08)', text_color='#6366f1', sort_order=3),
                    AIModel(name='Llama 3.3 70B', model_id='meta-llama/llama-3.3-70b-instruct', provider='meta',
                           description='نموذج مفتوح المصدر قوي، ممتاز في المحادثات متعددة اللغات والمهام العامة',
                           icon='bi-cpu', color='rgba(14,165,233,0.08)', text_color='#0ea5e9', sort_order=4),
                    AIModel(name='Gemma 4 31B', model_id='google/gemma-4-31b-it', provider='google',
                           description='نموذج متعدد الوسائط من جوجل، يدعم فهم الصور والنصوص معاً بأكثر من 140 لغة',
                           icon='bi-image', color='rgba(239,68,68,0.08)', text_color='#ef4444', sort_order=5),
                    AIModel(name='Grok 4.3', model_id='x-ai/grok-4.3', provider='xai',
                           description='نموذج متخصص في إعادة صياغة الطلبات بطريقة يفهمها النماذج الأخرى لتعزيز دقة الاستجابات',
                           icon='bi-magic', color='rgba(168,85,247,0.08)', text_color='#a855f7',
                           is_rewriter=True, sort_order=6),
                ]
                for m in default_models:
                    db.session.add(m)
                db.session.commit()
            _db_initialized = True
            logger.info("Database initialized.")
        except Exception as e:
            logger.error(f"DB init error: {e}")
            db.session.rollback()

# ---------- Public routes ----------
@app.route('/')
def index():
    site = SiteSetting.query.first()
    if site and site.status == 'off':
        return render_template('index.html', site_status='off', offline_message=site.offline_message)

    categories = Category.query.order_by(Category.sort_order).all()
    library_items = PromptLibrary.query.order_by(PromptLibrary.created_at.desc()).all()
    active_ad = LibraryAd.query.filter_by(is_active=True).order_by(LibraryAd.created_at.desc()).first()

    ad_dict = None
    if active_ad:
        ad_dict = {
            'id': active_ad.id,
            'title': active_ad.title,
            'text': active_ad.text,
            'image_url': active_ad.image_url,
            'button_text': active_ad.button_text,
            'button_link': active_ad.button_link,
            'duration_seconds': active_ad.duration_seconds
        }

    return render_template('index.html',
                           categories=categories,
                           library_items=library_items,
                           active_ad=ad_dict,
                           site_status='on')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'بيانات غير صحيحة'}), 400

        title = data.get('title', '').strip()
        category = data.get('category', 'general').strip()
        prompt_text = data.get('prompt_text', '').strip()
        image_url = data.get('image_url', '').strip()
        publisher_name = data.get('publisher_name', '').strip()
        csrf_token = data.get('csrf_token', '')

        if not validate_csrf_token(csrf_token):
            return jsonify({'success': False, 'message': 'CSRF خطأ'}), 400

        if not title or not prompt_text:
            return jsonify({'success': False, 'message': 'يرجى ملء العنوان ونص البرومبت'}), 400

        try:
            contribution = UploadContribution(
                title=title,
                category=category,
                prompt_text=prompt_text,
                image_url=image_url or None,
                publisher_name=publisher_name or None
            )
            db.session.add(contribution)
            db.session.commit()
            return jsonify({'success': True, 'message': 'تم استلام مساهمتك بنجاح! سيتم مراجعتها قريباً.'})
        except Exception as e:
            db.session.rollback()
            logger.error(f"Upload error: {e}")
            return jsonify({'success': False, 'message': 'خطأ في حفظ البيانات'}), 500

    categories = Category.query.order_by(Category.sort_order).all()
    return render_template('upload.html', categories=categories, csrf_token=generate_csrf_token())

# ==================== NEW: Chat Routes ====================
@app.route('/chat')
def chat_page():
    return render_template('chat.html')

@app.route('/api/models')
def get_models():
    """Get all active AI models"""
    models = AIModel.query.filter_by(is_active=True).order_by(AIModel.sort_order).all()
    return jsonify({
        'success': True,
        'models': [{
            'id': m.id,
            'name': m.name,
            'model_id': m.model_id,
            'provider': m.provider,
            'description': m.description,
            'icon': m.icon,
            'color': m.color,
            'text_color': m.text_color,
            'is_rewriter': m.is_rewriter
        } for m in models]
    })

@app.route('/api/threads')
def get_threads():
    """Get all chat threads for current session"""
    session_id = get_session_id()
    threads = ChatThread.query.filter_by(session_id=session_id).order_by(ChatThread.updated_at.desc()).all()
    return jsonify({
        'success': True,
        'threads': [{
            'id': t.id,
            'title': t.title,
            'model_id': t.model_id,
            'model_name': t.model.name if t.model else None,
            'message_count': len(t.messages),
            'updated_at': t.updated_at.isoformat() if t.updated_at else None
        } for t in threads]
    })

@app.route('/api/threads/<int:thread_id>')
def get_thread(thread_id):
    """Get a specific thread with all messages"""
    session_id = get_session_id()
    thread = ChatThread.query.filter_by(id=thread_id, session_id=session_id).first_or_404()
    messages = ChatMessage.query.filter_by(thread_id=thread_id).order_by(ChatMessage.created_at).all()
    return jsonify({
        'success': True,
        'thread': {
            'id': thread.id,
            'title': thread.title,
            'model_id': thread.model_id,
            'model_name': thread.model.name if thread.model else None,
            'created_at': thread.created_at.isoformat() if thread.created_at else None
        },
        'messages': [{
            'id': m.id,
            'role': m.role,
            'content': m.content,
            'created_at': m.created_at.isoformat() if m.created_at else None
        } for m in messages]
    })

@app.route('/api/threads/<int:thread_id>', methods=['DELETE'])
def delete_thread(thread_id):
    """Delete a chat thread"""
    session_id = get_session_id()
    thread = ChatThread.query.filter_by(id=thread_id, session_id=session_id).first_or_404()
    try:
        ChatMessage.query.filter_by(thread_id=thread_id).delete()
        db.session.delete(thread)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Delete thread error: {e}")
        return jsonify({'success': False, 'message': 'خطأ في الحذف'}), 500

@app.route('/api/threads/<int:thread_id>/clear', methods=['POST'])
def clear_thread(thread_id):
    """Clear all messages in a thread"""
    session_id = get_session_id()
    thread = ChatThread.query.filter_by(id=thread_id, session_id=session_id).first_or_404()
    try:
        ChatMessage.query.filter_by(thread_id=thread_id).delete()
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Clear thread error: {e}")
        return jsonify({'success': False, 'message': 'خطأ'}), 500

@app.route('/api/chat', methods=['POST'])
def chat_stream():
    """Stream chat response from AI model"""
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'No data'}), 400

    user_message = data.get('message', '').strip()
    model_id = data.get('model_id')
    thread_id = data.get('thread_id')

    if not user_message:
        return jsonify({'success': False, 'message': 'Empty message'}), 400

    # Get model
    model = AIModel.query.get(model_id)
    if not model or not model.is_active:
        return jsonify({'success': False, 'message': 'Model not found'}), 404

    session_id = get_session_id()

    # Get or create thread
    thread = None
    if thread_id:
        thread = ChatThread.query.filter_by(id=thread_id, session_id=session_id).first()

    if not thread:
        # Generate title from first message
        title = user_message[:50] + '...' if len(user_message) > 50 else user_message
        thread = ChatThread(
            session_id=session_id,
            title=title,
            model_id=model.id
        )
        db.session.add(thread)
        db.session.commit()
        thread_id = thread.id

    # Update thread model if changed
    if thread.model_id != model.id:
        thread.model_id = model.id

    # Save user message
    user_msg = ChatMessage(thread_id=thread.id, role='user', content=user_message)
    db.session.add(user_msg)
    db.session.commit()

    # Update thread timestamp
    thread.updated_at = datetime.utcnow()
    db.session.commit()

    # Get conversation history
    history = ChatMessage.query.filter_by(thread_id=thread.id).order_by(ChatMessage.created_at).all()
    messages_for_api = []
    for msg in history[-20:]:  # Keep last 20 messages for context
        messages_for_api.append({
            'role': msg.role,
            'content': msg.content
        })

    # If model is a rewriter (like grok-4.3), rewrite the prompt first
    final_messages = messages_for_api.copy()
    if model.is_rewriter:
        # For rewriter models, we send a system prompt to rewrite the user's request
        rewrite_system = "أنت مساعد متخصص في إعادة صياغة الطلبات. مهمتك هي إعادة صياغة طلب المستخدم بطريقة أكثر وضوحاً وتفصيلاً يفهمها النماذج الأخرى بشكل أفضل. حافظ على المعنى الأصلي وأضف تفاصيل سياقية. اكتب فقط النص المعاد صياغته بدون أي تعليقات إضافية."
        rewrite_messages = [
            {'role': 'system', 'content': rewrite_system},
            {'role': 'user', 'content': f"أعد صياغة هذا الطلب بشكل أفضل: {user_message}"}
        ]
        try:
            rewritten = call_openrouter(rewrite_messages, model.model_id)
            if rewritten:
                # Replace the last user message with rewritten version
                final_messages[-1]['content'] = rewritten.strip()
        except Exception as e:
            logger.error(f"Rewrite error: {e}")
            # Fall back to original message

    def generate():
        full_response = ""
        try:
            # Call OpenRouter API with streaming
            api_key = Config.OPENROUTER_API_KEY
            if not api_key:
                yield f"data: {json.dumps({'error': 'API key not configured'})}\n\n"
                return

            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': request.headers.get('Referer', ''),
                'X-Title': 'UFOQ Chat'
            }

            payload = {
                'model': model.model_id,
                'messages': final_messages,
                'stream': True
            }

            response = requests.post(
                'https://openrouter.ai/api/v1/chat/completions',
                headers=headers,
                json=payload,
                stream=True,
                timeout=120
            )

            if response.status_code != 200:
                error_text = response.text
                logger.error(f"OpenRouter error: {response.status_code} - {error_text}")
                yield f"data: {json.dumps({'error': f'API error: {response.status_code}'})}\n\n"
                return

            # Send thread ID first
            yield f"data: {json.dumps({'thread_id': thread.id})}\n\n"

            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get('choices', [{}])[0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            full_response += content
                            yield f"data: {json.dumps({'content': content})}\n\n"
                    except json.JSONDecodeError:
                        continue

            # Save assistant message
            assistant_msg = ChatMessage(
                thread_id=thread.id,
                role='assistant',
                content=full_response
            )
            db.session.add(assistant_msg)
            db.session.commit()

            # Update thread title if it's the first exchange
            if len(history) <= 2:
                # Generate a better title
                thread.title = generate_thread_title(user_message)
                db.session.commit()

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Chat stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

def call_openrouter(messages, model_id):
    """Non-streaming call to OpenRouter"""
    api_key = Config.OPENROUTER_API_KEY
    if not api_key:
        return None

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': request.headers.get('Referer', ''),
        'X-Title': 'UFOQ Chat'
    }

    payload = {
        'model': model_id,
        'messages': messages,
        'stream': False
    }

    response = requests.post(
        'https://openrouter.ai/api/v1/chat/completions',
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code == 200:
        data = response.json()
        return data.get('choices', [{}])[0].get('message', {}).get('content', '')
    return None

def generate_thread_title(first_message):
    """Generate a concise title from the first message"""
    # Simple extraction - take first 30 chars or first sentence
    title = first_message.strip()
    if len(title) > 40:
        title = title[:37] + '...'
    return title

# ---------- Admin ----------
@app.route('/admin', methods=['GET', 'POST'])
def admin_panel():
    if request.method == 'POST' and request.form.get('password') == Config.ADMIN_PASSWORD:
        session['logged_in'] = True
        return redirect(url_for('admin_panel'))

    if session.get('logged_in'):
        categories = Category.query.order_by(Category.sort_order).all()
        library_items = PromptLibrary.query.order_by(PromptLibrary.created_at.desc()).all()
        library_ads = LibraryAd.query.order_by(LibraryAd.created_at.desc()).all()
        site_settings = SiteSetting.query.first()
        contributions = UploadContribution.query.order_by(UploadContribution.created_at.desc()).all()
        ai_models = AIModel.query.order_by(AIModel.sort_order).all()
        return render_template('admin.html',
                               categories=categories,
                               library_items=library_items,
                               library_ads=library_ads,
                               site_settings=site_settings,
                               contributions=contributions,
                               ai_models=ai_models,
                               csrf_token=generate_csrf_token())
    return render_template('admin.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('logged_in', None)
    return redirect(url_for('admin_panel'))

# ---------- Admin: Categories ----------
@app.route('/admin/category/add', methods=['POST'])
@admin_required
def add_category():
    if not validate_csrf_token(request.form.get('csrf_token')):
        return "CSRF Error", 400
    try:
        name = request.form.get('name', '').strip().lower().replace(' ', '_')
        display_name = request.form.get('display_name', '').strip()
        icon = request.form.get('icon', 'bi-tag').strip()
        sort_order = int(request.form.get('sort_order', 0))
        if not name or not display_name:
            flash('اسم التصنيف واسم العرض مطلوبان', 'error')
            return redirect(url_for('admin_panel'))
        if Category.query.filter_by(name=name).first():
            flash('التصنيف موجود مسبقاً', 'error')
            return redirect(url_for('admin_panel'))
        cat = Category(name=name, display_name=display_name, icon=icon, sort_order=sort_order)
        db.session.add(cat)
        db.session.commit()
        flash('تمت إضافة التصنيف', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error adding category: {e}")
        flash('خطأ في إضافة التصنيف', 'error')
    return redirect(url_for('admin_panel'))

@app.route('/admin/category/<int:category_id>/delete', methods=['POST'])
@admin_required
def delete_category(category_id):
    if not validate_csrf_token(request.form.get('csrf_token')):
        return "CSRF Error", 400
    try:
        cat = Category.query.get_or_404(category_id)
        PromptLibrary.query.filter_by(category=cat.name).update({'category': 'general'})
        db.session.delete(cat)
        db.session.commit()
        flash('تم حذف التصنيف', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting category: {e}")
        flash('خطأ في حذف التصنيف', 'error')
    return redirect(url_for('admin_panel'))

# ---------- Admin: Library ----------
@app.route('/admin/library/add', methods=['POST'])
@admin_required
def add_library_item():
    if not validate_csrf_token(request.form.get('csrf_token')):
        return "CSRF Error", 400
    try:
        item = PromptLibrary(
            title=request.form.get('title'),
            category=request.form.get('category', 'general'),
            image_url=request.form.get('image_url'),
            prompt_text=request.form.get('prompt_text'),
            publisher=request.form.get('publisher', '').strip() or None
        )
        db.session.add(item)
        db.session.commit()
        flash('تمت إضافة البرومبت', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error adding library item: {e}")
        flash('خطأ في إضافة البرومبت', 'error')
    return redirect(url_for('admin_panel'))

@app.route('/admin/library/<int:item_id>/delete', methods=['POST'])
@admin_required
def delete_library_item(item_id):
    if not validate_csrf_token(request.form.get('csrf_token')):
        return "CSRF Error", 400
    item = PromptLibrary.query.get_or_404(item_id)
    db.session.delete(item)
    db.session.commit()
    flash('تم حذف البرومبت', 'success')
    return redirect(url_for('admin_panel'))

@app.route('/admin/library/<int:item_id>/update', methods=['POST'])
@admin_required
def update_library_item(item_id):
    if not validate_csrf_token(request.form.get('csrf_token')):
        return "CSRF Error", 400
    item = PromptLibrary.query.get_or_404(item_id)
    item.title = request.form.get('title', item.title)
    item.category = request.form.get('category', item.category)
    item.image_url = request.form.get('image_url', item.image_url)
    item.prompt_text = request.form.get('prompt_text', item.prompt_text)
    item.publisher = request.form.get('publisher', item.publisher) or None
    db.session.commit()
    flash('تم تحديث البرومبت', 'success')
    return redirect(url_for('admin_panel'))

# ---------- Admin: Contributions ----------
@app.route('/admin/contribution/<int:contrib_id>/approve', methods=['POST'])
@admin_required
def approve_contribution(contrib_id):
    if not validate_csrf_token(request.form.get('csrf_token')):
        return "CSRF Error", 400
    try:
        contrib = UploadContribution.query.get_or_404(contrib_id)
        item = PromptLibrary(
            title=contrib.title,
            category=contrib.category,
            image_url=contrib.image_url or '',
            prompt_text=contrib.prompt_text,
            publisher=contrib.publisher_name
        )
        db.session.add(item)
        contrib.status = 'approved'
        db.session.commit()
        flash('تمت الموافقة على المساهمة', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error approving contribution: {e}")
        flash('خطأ', 'error')
    return redirect(url_for('admin_panel'))

@app.route('/admin/contribution/<int:contrib_id>/reject', methods=['POST'])
@admin_required
def reject_contribution(contrib_id):
    if not validate_csrf_token(request.form.get('csrf_token')):
        return "CSRF Error", 400
    try:
        contrib = UploadContribution.query.get_or_404(contrib_id)
        contrib.status = 'rejected'
        db.session.commit()
        flash('تم رفض المساهمة', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error rejecting contribution: {e}")
        flash('خطأ', 'error')
    return redirect(url_for('admin_panel'))

@app.route('/admin/contribution/<int:contrib_id>/delete', methods=['POST'])
@admin_required
def delete_contribution(contrib_id):
    if not validate_csrf_token(request.form.get('csrf_token')):
        return "CSRF Error", 400
    try:
        contrib = UploadContribution.query.get_or_404(contrib_id)
        db.session.delete(contrib)
        db.session.commit()
        flash('تم حذف المساهمة', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting contribution: {e}")
        flash('خطأ', 'error')
    return redirect(url_for('admin_panel'))

# ---------- Admin: Library Ads ----------
@app.route('/admin/library_ad/add', methods=['POST'])
@admin_required
def add_library_ad():
    if not validate_csrf_token(request.form.get('csrf_token')):
        return "CSRF Error", 400
    try:
        ad = LibraryAd(
            title=request.form.get('title'),
            text=request.form.get('text'),
            image_url=request.form.get('image_url') or None,
            button_text=request.form.get('button_text'),
            button_link=request.form.get('button_link'),
            duration_seconds=int(request.form.get('duration_seconds', 5)),
            is_active=request.form.get('is_active') == 'on'
        )
        db.session.add(ad)
        db.session.commit()
        flash('تمت إضافة الإعلان', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error adding library ad: {e}")
        flash('خطأ في إضافة الإعلان', 'error')
    return redirect(url_for('admin_panel'))

@app.route('/admin/library_ad/<int:ad_id>/delete', methods=['POST'])
@admin_required
def delete_library_ad(ad_id):
    if not validate_csrf_token(request.form.get('csrf_token')):
        return "CSRF Error", 400
    try:
        ad = LibraryAd.query.get_or_404(ad_id)
        db.session.delete(ad)
        db.session.commit()
        flash('تم حذف الإعلان', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting library ad: {e}")
        flash('خطأ في حذف الإعلان', 'error')
    return redirect(url_for('admin_panel'))

@app.route('/admin/library_ad/<int:ad_id>/toggle', methods=['POST'])
@admin_required
def toggle_library_ad(ad_id):
    if not validate_csrf_token(request.form.get('csrf_token')):
        return "CSRF Error", 400
    try:
        ad = LibraryAd.query.get_or_404(ad_id)
        ad.is_active = not ad.is_active
        db.session.commit()
        flash('تم تغيير حالة الإعلان', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error toggling library ad: {e}")
        flash('خطأ في تغيير حالة الإعلان', 'error')
    return redirect(url_for('admin_panel'))

# ==================== NEW: Admin AI Models Management ====================
@app.route('/admin/ai_model/add', methods=['POST'])
@admin_required
def add_ai_model():
    if not validate_csrf_token(request.form.get('csrf_token')):
        return "CSRF Error", 400
    try:
        model = AIModel(
            name=request.form.get('name'),
            model_id=request.form.get('model_id'),
            provider=request.form.get('provider'),
            description=request.form.get('description'),
            icon=request.form.get('icon', 'bi-cpu'),
            color=request.form.get('color', 'rgba(14,165,233,0.08)'),
            text_color=request.form.get('text_color', '#0ea5e9'),
            is_active=request.form.get('is_active') == 'on',
            is_rewriter=request.form.get('is_rewriter') == 'on',
            sort_order=int(request.form.get('sort_order', 0))
        )
        db.session.add(model)
        db.session.commit()
        flash('تمت إضافة النموذج', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error adding AI model: {e}")
        flash('خطأ في إضافة النموذج', 'error')
    return redirect(url_for('admin_panel'))

@app.route('/admin/ai_model/<int:model_id>/update', methods=['POST'])
@admin_required
def update_ai_model(model_id):
    if not validate_csrf_token(request.form.get('csrf_token')):
        return "CSRF Error", 400
    try:
        model = AIModel.query.get_or_404(model_id)
        model.name = request.form.get('name', model.name)
        model.model_id = request.form.get('model_id', model.model_id)
        model.provider = request.form.get('provider', model.provider)
        model.description = request.form.get('description', model.description)
        model.icon = request.form.get('icon', model.icon)
        model.color = request.form.get('color', model.color)
        model.text_color = request.form.get('text_color', model.text_color)
        model.is_active = request.form.get('is_active') == 'on'
        model.is_rewriter = request.form.get('is_rewriter') == 'on'
        model.sort_order = int(request.form.get('sort_order', model.sort_order))
        db.session.commit()
        flash('تم تحديث النموذج', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating AI model: {e}")
        flash('خطأ في تحديث النموذج', 'error')
    return redirect(url_for('admin_panel'))

@app.route('/admin/ai_model/<int:model_id>/delete', methods=['POST'])
@admin_required
def delete_ai_model(model_id):
    if not validate_csrf_token(request.form.get('csrf_token')):
        return "CSRF Error", 400
    try:
        model = AIModel.query.get_or_404(model_id)
        db.session.delete(model)
        db.session.commit()
        flash('تم حذف النموذج', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting AI model: {e}")
        flash('خطأ في حذف النموذج', 'error')
    return redirect(url_for('admin_panel'))

@app.route('/admin/ai_model/<int:model_id>/toggle', methods=['POST'])
@admin_required
def toggle_ai_model(model_id):
    if not validate_csrf_token(request.form.get('csrf_token')):
        return "CSRF Error", 400
    try:
        model = AIModel.query.get_or_404(model_id)
        model.is_active = not model.is_active
        db.session.commit()
        flash('تم تغيير حالة النموذج', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error toggling AI model: {e}")
        flash('خطأ', 'error')
    return redirect(url_for('admin_panel'))

# ---------- Admin: Site Settings ----------
@app.route('/api/admin/update_site_settings', methods=['POST'])
@admin_required
def update_site_settings():
    if not validate_csrf_token(request.json.get('csrf_token')):
        return jsonify({'error': 'CSRF Error'}), 400
    s = SiteSetting.query.first()
    s.status = request.json.get('status', 'on')
    s.offline_message = request.json.get('offline_message', 'الموقع تحت الصيانة حالياً.')
    db.session.commit()
    return jsonify({'success': True})

@app.route('/api/admin/get_site_status')
@admin_required
def get_site_status():
    s = SiteSetting.query.first()
    return jsonify({
        'status': s.status if s else 'on',
        'offline_message': s.offline_message if s else 'الموقع تحت الصيانة حالياً.'
    })

# ---------- Health ----------
@app.route('/health')
def health_check():
    return jsonify({'status': 'ok', 'timestamp': datetime.utcnow().isoformat()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
