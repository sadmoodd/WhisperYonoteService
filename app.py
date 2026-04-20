from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Header, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import whisper
import os
import time
import tempfile
import requests
import logging
from typing import Optional, List, Dict
from pydantic import BaseModel
from pydub import AudioSegment
from dotenv import load_dotenv
import torch
import uvicorn
import gc
import json
import re
import subprocess
import shutil
import asyncio

logging.basicConfig(level=logging.INFO, filename="whisper.log",
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Whisper AI API", version="2.0")

import base64
class TokenCrypto:
    """Класс для шифрования/расшифровки токенов"""
    @staticmethod
    def decrypt(encrypted_text: str) -> Optional[str]:
        if not encrypted_text:
            return None
        try:
            decoded = base64.b64decode(encrypted_text).decode('latin1')
            key = os.getenv("SECRET_KEY")
            result = ''
            for i in range(len(decoded)):
                char_code = ord(decoded[i]) ^ ord(key[i % len(key)])
                result += chr(char_code)
            return result   
        except Exception as e:
            logger.error(f"Token decryption failed: {e}")
            return None
    @staticmethod
    def encrypt(text: str) -> Optional[str]:
        if not text:
            return None  
        try:
            key = os.getenv("SECRET_KEY")
            result = ''
            for i in range(len(text)):
                char_code = ord(text[i]) ^ ord(key[i % len(key)])
                result += chr(char_code)
            return base64.b64encode(result.encode('latin1')).decode()
        except Exception as e:
            logger.error(f"Token encryption failed: {e}")
            return None

def get_token_from_request(request: Request, form_data: Optional[Dict] = None) -> Optional[str]:
    """
    Извлекает токен из разных частей запроса
    """
    # Проверяем заголовок X-Encrypted-Token
    encrypted_token = request.headers.get('yonote_token')
    if encrypted_token:
        decrypted = TokenCrypto.decrypt(encrypted_token)
        if decrypted:
            return decrypted
    
    # Проверяем form data
    if form_data:
        encrypted_token = form_data.get('token')
        if encrypted_token:
            decrypted = TokenCrypto.decrypt(encrypted_token)
            if decrypted:
                return decrypted
    
    # Проверяем Authorization заголовок (для обратной совместимости)
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.replace('Bearer ', '')
        # Пробуем расшифровать
        decrypted = TokenCrypto.decrypt(token)
        return decrypted if decrypted else token
    
    return None


# Модели данных
class Task(BaseModel):
    title: str
    description: Optional[str] = None
    assignee: Optional[str] = None
    priority: Optional[str] = "medium"
    deadline: Optional[str] = None
    estimated_hours: Optional[float] = None

class Project(BaseModel):
    name: str
    description: Optional[str] = None
    tasks: List[Task]
    
class YonoteRequest(BaseModel):
    project: Project
    notes: Optional[str] = None
    yonote_space_id: Optional[str] = None

API_URL = f"{os.getenv('BASE_URL')}"
headers = {
    "Authorization": f"Basic {os.getenv('API_KEY')}",
}

def query(payload):
    """Безопасный запрос к LLM"""
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        return data["message"]['content'].strip()
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        return "Ошибка суммаризации"

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Очистка памяти перед загрузкой модели
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

# Устанавливаем переменные окружения для оптимизации памяти
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Загрузка модели
if torch.cuda.is_available():
    logger.info("CUDA доступна, загружаем модель base...")
    base_model = whisper.load_model("base")
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(base_model)
        logger.info(f"✅ Модель распределена на {torch.cuda.device_count()} GPU")
    else:
        model = base_model
        logger.info("✅ Модель загружена на 1 GPU")
    
    model = model.to('cuda')
    
    for i in range(torch.cuda.device_count()):
        memory = torch.cuda.memory_allocated(i) / 1024**3
        logger.info(f"GPU {i}: {memory:.2f} ГБ")
else:
    model = whisper.load_model("base")
    logger.info("⚠️ CUDA не найдена, используем CPU")

async def transcribe_chunk_async(chunk_path: str, chunk_idx: int):
    """Безопасная транскрибация чанка"""
    try:
        if hasattr(model, 'module'):
            result = model.module.transcribe(chunk_path)
        else:
            result = model.transcribe(chunk_path)
        
        torch.cuda.empty_cache()
        
        return {
            "text": result["text"].strip(),
            "segments": len(result.get("segments", []))
        }
    except Exception as e:
        logger.error(f"Chunk {chunk_idx} error: {e}")
        torch.cuda.empty_cache()
        return {"text": "", "segments": 0}

async def extract_audio_from_video(video_path: str, output_audio_path: str) -> bool:
    """
    Извлекает аудио из видеофайла с помощью ffmpeg
    Возвращает True если успешно, False если ошибка
    """
    try:
        # Проверяем наличие ffmpeg
        if not shutil.which('ffmpeg'):
            logger.error("ffmpeg не установлен. Установите: sudo apt install ffmpeg")
            return False
        
        # Команда для извлечения аудио
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            output_audio_path
        ]
        
        logger.info(f"Извлечение аудио из видео: {video_path}")
        
        # Запускаем ffmpeg
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"ffmpeg error: {stderr.decode()}")
            return False
        
        logger.info(f"Аудио успешно извлечено: {output_audio_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error extracting audio from video: {e}")
        return False

# Основной эндпоинт транскрипции
@app.post("/whisper/api/transcribe")
async def transcribe_audio(file: UploadFile = File(None), audio: UploadFile = File(None)):
    upload_file = file or audio
    if not upload_file:
        raise HTTPException(status_code=400, detail="No file uploaded (field 'file' or 'audio' expected)")
    
    logger.info(f"Запрос получен - файл: {upload_file.filename}")
    start = time.time()
    
    if not upload_file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    if upload_file.size > 250 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 250MB)")
    
    original_filename = upload_file.filename
    file_ext = original_filename.split('.')[-1].lower()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_filepath = os.path.join(tmpdir, f"original.{file_ext}")
        
        content = await upload_file.read()
        with open(original_filepath, "wb") as f:
            f.write(content)
        
        file_size = os.path.getsize(original_filepath)
        
        try:
            audio_segment = AudioSegment.from_file(original_filepath)
            chunk_length_ms = 60 * 1000
            chunks = [audio_segment[i:i + chunk_length_ms] for i in range(0, len(audio_segment), chunk_length_ms)]
            
            logger.info(f"🎯 {len(chunks)} чанков по 60s")
            
            chunk_paths = []
            for idx, chunk in enumerate(chunks):
                chunk_path = os.path.join(tmpdir, f"chunk_{idx}.wav")
                chunk.export(chunk_path, format="wav")
                chunk_paths.append(chunk_path)
            
            all_results = []
            for idx, path in enumerate(chunk_paths):
                result = await transcribe_chunk_async(path, idx)
                logger.info(f"Чанк {idx + 1} из {len(chunk_paths)} обработан")
                all_results.append(result)
            
            logger.info("Все чанки обработаны успешно")
            all_transcriptions = [r["text"] for r in all_results if r["text"]]
            total_segments = sum(r["segments"] for r in all_results)
            
            full_transcription = " ".join(all_transcriptions).strip()
            logger.info("Транскрипция получена")
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise HTTPException(status_code=500, detail="Audio processing error")
    
    total_time = time.time() - start
    words = len(full_transcription.split())
    words_per_second = round(words / total_time, 2) if total_time > 0 else 0
    
    if full_transcription:
        prompt = f"""Проанализируй текст из аудио разбитый на чанки по 60с. Верни summary и примерные таймкоды и попытайся выделить собеседников,
        если это возможно. Если ничего не смог найти верни Не удалось аннотировать текст

        Транскрипция: {full_transcription}

        Только текст:"""
                
        payload = {
            "model": os.getenv("MODEL_NAME", 'gemma3:1b'),
            "messages": [
                {"role": "user", "content": "Ты анализатор аудио-текстов. Дай summary + таймкоды." + prompt}
            ],
            "stream": False
        }
        logger.info('Обратились к LLM за аннотацией')
        summary = query(payload)
        logger.info(f'SUMMARY - {summary}')
    else:
        logger.warning("Модель не распознала текст, возвращаем 'Нет распознанного текста'")
        summary = "Нет распознанного текста"
    
    return {
        "success": True,
        "transcription": full_transcription,
        "summary": summary,
        "stats": {
            "total_processing_time": round(total_time, 2),
            "words_per_second": words_per_second,
            "file_size_mb": round(file_size / (1024*1024), 1),
            "chunks_processed": len(chunk_paths),
            "segments_count": total_segments,
            "total_words": words,
        },
        "filename": original_filename,
        "processing_time": round(total_time, 2)
    }

def extract_project_and_tasks_from_transcription(transcription: str) -> dict:
    """Извлекает информацию о проекте и задачах из транскрипции"""
    
    prompt = f"""
Ты - интеллектуальный помощник по управлению проектами. Проанализируй транскрипцию аудио и извлеки из неё:

1. **ПРОЕКТ**: 
   - Название проекта
   - Краткое описание/цель
   - Ключевые метрики (если упоминаются)

2. **ЗАДАЧИ**: 
   Для каждой задачи определи:
   - Название задачи (кратко и ёмко)
   - Описание (подробности из разговора)
   - Исполнитель (кто назначен, если указано)
   - Приоритет (high/medium/low на основе срочности и важности)
   - Дедлайн (если упоминается дата)
   - Оценка времени (часы/дни, если есть)

3. **КОНТЕКСТ**:
   - Важные обсуждения
   - Зависимости между задачами
   - Риски и блокеры
   - Решения, принятые в ходе разговора

Транскрипция:
{transcription}

Верни ответ строго в формате JSON со следующей структурой:
{{
    "project": {{
        "name": "название проекта",
        "description": "описание проекта",
        "key_metrics": "ключевые метрики (если есть)"
    }},
    "tasks": [
        {{
            "title": "название задачи",
            "description": "подробное описание",
            "assignee": "исполнитель (если указан)",
            "priority": "high/medium/low",
            "deadline": "дата дедлайна (если есть)",
            "estimated_hours": "оценка времени в часах (если есть)"
        }}
    ],
    "context": {{
        "dependencies": ["зависимость 1", "зависимость 2"],
        "risks": ["риск 1", "риск 2"],
        "decisions": ["решение 1", "решение 2"],
        "important_notes": "важные заметки"
    }}
}}

Если какая-то информация отсутствует, оставь поле пустым или null.
"""
    
    payload = {
        "model": os.getenv("MODEL_NAME", 'gemma3:1b'),
        "messages": [
            {"role": "system", "content": "Ты эксперт по управлению проектами. Отвечай только JSON."},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "temperature": 0.3
    }
    
    try:
        logger.info('Запрос к LLM для извлечения проекта и задач')
        response_text = query(payload)
        
        json_match = re.search(r'\{.*\}(?=\s*$|\s*\Z)', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group()
        
        result = json.loads(response_text)
        logger.info(f'Успешно извлечено: проект "{result.get("project", {}).get("name", "не указан")}", {len(result.get("tasks", []))} задач')
        
        if not result.get("tasks"):
            result["tasks"] = []
            if not result.get("context", {}).get("important_notes"):
                result.setdefault("context", {})["important_notes"] = "В транскрипции не удалось выделить конкретные задачи"
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f'Ошибка парсинга JSON: {e}, ответ: {response_text}')
        return {
            "project": {
                "name": "Не удалось определить",
                "description": "Ошибка обработки"
            },
            "tasks": [],
            "context": {
                "dependencies": [],
                "risks": [],
                "decisions": [],
                "important_notes": "Не удалось структурировать данные"
            }
        }
    except Exception as e:
        logger.error(f'Ошибка при извлечении данных: {e}')
        return {
            "project": {
                "name": "Ошибка",
                "description": str(e)
            },
            "tasks": [],
            "context": {
                "important_notes": "Ошибка обработки транскрипции"
            }
        }

def format_for_yonote(project_data: dict, original_transcription: str = None) -> dict:
    """Форматирует данные для отправки в Yonote"""
    yonote_content = {
        "title": project_data.get("project", {}).get("name", "Новый проект"),
        "description": project_data.get("project", {}).get("description", ""),
        "content": {
            "project_info": {
                "name": project_data.get("project", {}).get("name", ""),
                "description": project_data.get("project", {}).get("description", ""),
                "key_metrics": project_data.get("project", {}).get("key_metrics", "")
            },
            "tasks": [],
            "context": project_data.get("context", {}),
            "original_transcription": original_transcription[:1000] if original_transcription else None
        }
    }
    
    for task in project_data.get("tasks", []):
        yonote_content["content"]["tasks"].append({
            "title": task.get("title", "Новая задача"),
            "description": task.get("description", ""),
            "assignee": task.get("assignee", "Не назначен"),
            "priority": task.get("priority", "medium"),
            "deadline": task.get("deadline", "Не указан"),
            "estimated_hours": task.get("estimated_hours", 0),
            "status": "open"
        })
    
    yonote_content["summary"] = f"""
📊 **Статистика проекта:**
- Всего задач: {len(project_data.get('tasks', []))}
- Приоритеты: high - {len([t for t in project_data.get('tasks', []) if t.get('priority') == 'high'])}, 
  medium - {len([t for t in project_data.get('tasks', []) if t.get('priority') == 'medium'])}, 
  low - {len([t for t in project_data.get('tasks', []) if t.get('priority') == 'low'])}
- Исполнители: {', '.join(set(t.get('assignee') for t in project_data.get('tasks', []) if t.get('assignee'))) or 'не определены'}

📝 **Ключевые решения:**
{chr(10).join(f'- {d}' for d in project_data.get('context', {}).get('decisions', [])) or '- Не указаны'}

⚠️ **Риски и блокеры:**
{chr(10).join(f'- {r}' for r in project_data.get('context', {}).get('risks', [])) or '- Не выявлены'}
"""
    
    return yonote_content

# Импорт клиента Yonote
from yonote_client import YonoteClient, format_project_to_markdown, get_yonote_client

# Создаем глобальный экземпляр клиента Yonote
yonote_client = get_yonote_client()

# ========== НОВЫЕ ФУНКЦИИ ДЛЯ ОПРЕДЕЛЕНИЯ КОМАНД ==========

def detect_command(transcription: str) -> tuple:
    """
    Определяет тип команды: create (создать) или update (изменить)
    Возвращает (command_type, project_name, changes_text)
    """
    transcription_lower = transcription.lower()
    
    # Ключевые слова для обновления
    update_keywords = ['измени', 'поменяй', 'обнови', 'добавь в проект', 'удали из проекта', 'исправь в проекте', 'отредактируй']
    create_keywords = ['создай', 'новый проект', 'запусти проект', 'сделай проект']
    
    is_update = any(kw in transcription_lower for kw in update_keywords)
    is_create = any(kw in transcription_lower for kw in create_keywords)
    
    # Если нет явных команд, считаем что это создание
    if not is_update and not is_create:
        is_create = True
    
    # Извлекаем название проекта
    project_name = None
    # Ищем фразы типа "в проекте [название]" или "проект [название]"
    name_match = re.search(r'(?:в проекте|проект(?:е)?)\s+["\']?([^"\'\n,]+)["\']?', transcription_lower)
    if name_match:
        project_name = name_match.group(1).strip()
    
    # Если не нашли по шаблону, пробуем взять слово после ключевых слов
    if not project_name and (is_update or is_create):
        words = transcription_lower.split()
        for i, word in enumerate(words):
            if word in update_keywords or word in create_keywords:
                if i + 1 < len(words):
                    # Пропускаем предлоги
                    if words[i + 1] in ['в', 'на', 'по', 'для'] and i + 2 < len(words):
                        project_name = words[i + 2]
                    else:
                        project_name = words[i + 1]
                    break
    
    return ("update" if is_update else "create", project_name, transcription)


def find_project_by_name(project_name: str, collection_id: str = None) -> Optional[str]:
    """
    Ищет документ в Yonote по названию проекта
    Возвращает document_id если найден
    """
    if not project_name:
        return None
    
    try:
        # Используем поиск через API
        search_result = yonote_client.search_documents(project_name, collection_id)
        
        if search_result and search_result.get("data") and len(search_result["data"]) > 0:
            # Ищем точное совпадение названия
            for doc in search_result["data"]:
                doc_title = doc.get("title", "").lower()
                if project_name.lower() in doc_title:
                    return doc.get("id")
            # Если нет точного, возвращаем первый
            return search_result["data"][0].get("id")
        
        return None
    except Exception as e:
        logger.error(f"Error finding project: {e}")
        return None


def extract_changes_from_voice(transcription: str, current_project: dict) -> dict:
    """
    Извлекает изменения из голосовой команды и возвращает обновленный проект
    """
    prompt = f"""
Ты - помощник по управлению проектами. Нужно обновить существующий проект на основе голосовых указаний.

ТЕКУЩИЙ ПРОЕКТ (JSON):
{json.dumps(current_project, ensure_ascii=False, indent=2)}

ГОЛОСОВЫЕ УКАЗАНИЯ:
{transcription}

Выполни изменения:
- Если указано изменить приоритет задачи: найди задачу и измени priority на high/medium/low
- Если указано добавить задачу: создай новую задачу с указанными полями
- Если указано удалить задачу: удали её
- Если указано изменить исполнителя: измени assignee
- Если указано изменить дедлайн: измени deadline
- Если указано добавить решение/риск: добавь в context

Верни ОБНОВЛЕННЫЙ проект в том же JSON формате, что и текущий.
Если задача не найдена, просто проигнорируй изменение.
Сохрани все остальные поля без изменений.
"""
    
    payload = {
        "model": os.getenv("MODEL_NAME", 'gemma3:1b'),
        "messages": [
            {"role": "system", "content": "Ты эксперт по управлению проектами. Отвечай только JSON."},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "temperature": 0.3
    }
    
    try:
        response_text = query(payload)
        json_match = re.search(r'\{.*\}(?=\s*$|\s*\Z)', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group()
        
        return json.loads(response_text)
    except Exception as e:
        logger.error(f"Error extracting changes: {e}")
        return current_project








# ========== ОСНОВНОЙ ЭНДПОИНТ С УМНОЙ ЛОГИКОЙ (ПОДДЕРЖКА ВИДЕО) ==========

@app.post("/yonote/api/process-project")
async def process_project(
    request: Request,
    file: UploadFile = File(...),
    send_to_yonote: bool = False,
    collection_id: Optional[str] = None,
    document_id: Optional[str] = None,
    api_key: Optional[str] = None
):
    """
    Умный эндпоинт: определяет создавать новый проект или обновлять существующий
    Поддерживает аудио (mp3, wav, m4a, ogg) и видео (mp4, avi, mov, mkv)
    """
    token = get_token_from_request(request)
    
    # Если токен не найден в заголовках, пробуем из формы
    if not token:
        token = TokenCrypto.decrypt(token)
    
    # Создаем клиент Yonote
    if token:
        logger.info("Using user-provided token")
        yonote_client = YonoteClient(api_key=token)
    else:
        logger.info("Using default token from .env")
        yonote_client = get_yonote_client()



    upload_file = file
    if not upload_file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    logger.info(f"Обработка файла: {upload_file.filename}")
    start = time.time()
    
    if not upload_file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    if upload_file.size > 500 * 1024 * 1024:  # 500MB для видео
        raise HTTPException(status_code=400, detail="File too large (max 500MB)")
    
    original_filename = upload_file.filename
    file_ext = original_filename.split('.')[-1].lower()
    
    # Список поддерживаемых форматов
    video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv', 'm4v']
    audio_extensions = ['mp3', 'wav', 'm4a', 'ogg', 'flac', 'aac']
    
    full_transcription = ""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_filepath = os.path.join(tmpdir, f"original.{file_ext}")
        
        content = await upload_file.read()
        with open(original_filepath, "wb") as f:
            f.write(content)
        
        try:
            audio_path = original_filepath
            
            # Если это видео - извлекаем аудио
            if file_ext in video_extensions:
                audio_path = os.path.join(tmpdir, "extracted_audio.wav")
                success = await extract_audio_from_video(original_filepath, audio_path)
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to extract audio from video")
                logger.info(f"Видео конвертировано в аудио: {original_filename}")
            
            # Загружаем аудио
            audio_segment = AudioSegment.from_file(audio_path)
            chunk_length_ms = 60 * 1000
            chunks = [audio_segment[i:i + chunk_length_ms] for i in range(0, len(audio_segment), chunk_length_ms)]
            
            logger.info(f"🎯 {len(chunks)} чанков по 60s")
            
            chunk_paths = []
            for idx, chunk in enumerate(chunks):
                chunk_path = os.path.join(tmpdir, f"chunk_{idx}.wav")
                chunk.export(chunk_path, format="wav")
                chunk_paths.append(chunk_path)
            
            all_results = []
            for idx, path in enumerate(chunk_paths):
                result = await transcribe_chunk_async(path, idx)
                all_results.append(result)
            
            all_transcriptions = [r["text"] for r in all_results if r["text"]]
            full_transcription = " ".join(all_transcriptions).strip()
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    # ========== АНАЛИЗ КОМАНДЫ ==========
    command_type, project_name, changes_text = detect_command(full_transcription)
    
    logger.info(f"Обнаружена команда: {command_type}, проект: {project_name}")
    
    # Если передан document_id явно, считаем что это обновление
    if document_id:
        command_type = "update"
    
    # ========== ОБНОВЛЕНИЕ СУЩЕСТВУЮЩЕГО ПРОЕКТА ==========
    if command_type == "update" and (document_id or project_name):
        try:
            target_document_id = document_id
            if not target_document_id and project_name:
                target_document_id = find_project_by_name(project_name, collection_id)
                
                if not target_document_id:
                    logger.warning(f"Проект '{project_name}' не найден, создаем новый")
                    command_type = "create"
            
            if target_document_id:
                current_doc = yonote_client.get_document(target_document_id)
                
                if current_doc.get("error"):
                    logger.error(f"Ошибка получения документа: {current_doc['error']}")
                else:
                    current_project_data = extract_project_and_tasks_from_transcription(
                        current_doc.get("text", "")[:3000]
                    )
                    
                    if not current_project_data.get("project", {}).get("name"):
                        current_project_data = {
                            "project": {"name": project_name or "Проект"},
                            "tasks": [],
                            "context": {}
                        }
                    
                    updated_project = extract_changes_from_voice(changes_text, current_project_data)
                    
                    new_markdown = format_project_to_markdown(updated_project)
                    update_result = yonote_client.update_document(
                        document_id=target_document_id,
                        text=new_markdown,
                        publish=True
                    )
                    
                    if not update_result.get("error"):
                        return {
                            "success": True,
                            "action": "updated",
                            "document_id": target_document_id,
                            "changes_applied": changes_text[:500],
                            "updated_project": updated_project,
                            "message": f"Проект '{updated_project.get('project', {}).get('name', project_name)}' успешно обновлен"
                        }
                    else:
                        logger.error(f"Ошибка обновления: {update_result.get('error')}")
                        command_type = "create"
        except Exception as e:
            logger.error(f"Ошибка обновления: {e}")
            command_type = "create"
    
    # ========== СОЗДАНИЕ НОВОГО ПРОЕКТА ==========
    if command_type == "create":
        project_data = extract_project_and_tasks_from_transcription(full_transcription)
        yonote_data = format_for_yonote(project_data, full_transcription)
        
        response = {
            "success": True,
            "action": "created",
            "file_type": "video" if file_ext in video_extensions else "audio",
            "transcription_length": len(full_transcription),
            "project": project_data.get("project", {}),
            "tasks_count": len(project_data.get("tasks", [])),
            "tasks": project_data.get("tasks", []),
            "context": project_data.get("context", {}),
            "yonote_data": yonote_data,
            "processing_time": round(time.time() - start, 2),
            "filename": original_filename
        }
        
        if send_to_yonote:
            markdown_content = format_project_to_markdown(project_data)
            title = project_data.get("project", {}).get("name", "Новый проект")
            
            yonote_result = yonote_client.create_document(
                title=title,
                text=markdown_content,
                collection_id=collection_id,
                publish=True
            )
            
            response["yonote_sent"] = not yonote_result.get("error")
            response["yonote_response"] = yonote_result
            response["document_id"] = yonote_result.get("id") if not yonote_result.get("error") else None
        
        return response
    
    # Если ничего не подошло
    return {
        "success": False,
        "error": "Не удалось определить команду",
        "transcription": full_transcription[:500]
    }



# ========== ОСТАЛЬНЫЕ ЭНДПОИНТЫ ==========

@app.post("/yonote/api/send-to-yonote")
async def send_to_yonote_endpoint(
    request: Request,
    collection_id: Optional[str] = None,
    publish: bool = True
):
    """Отправляет данные проекта в Yonote"""
    data = await request.json()
    project_data = data.get("project_data")
    collection_id = data.get("collection_id")

    token = get_token_from_request(request)
    
    # Если токен не найден в заголовках, пробуем из формы
    if not token and yonote_token:
        token = TokenCrypto.decrypt(yonote_token)
    
    # Создаем клиент Yonote
    if token:
        logger.info("Using user-provided token")
        yonote_client = YonoteClient(api_key=token)
    else:
        logger.info("Using default token from .env")
        yonote_client = get_yonote_client()
    
    if not project_data:
        raise HTTPException(status_code=400, detail="No project_data provided")
    
    markdown_content = format_project_to_markdown(project_data)
    title = project_data.get("project", {}).get("name", "Новый проект")
    result = yonote_client.create_document(
        title=title,
        text=markdown_content,
        collection_id=collection_id,
        publish=publish
    )
    
    if result.get("error"):
        raise HTTPException(status_code=500, detail=f"Yonote API error: {result['error']}")
    
    return {
        "success": True,
        "message": f"Документ '{title}' успешно создан в Yonote",
        "document": result
    }


@app.get("/yonote/api/collections")
async def get_collections():
    """Получает список доступных коллекций в Yonote"""
    
    # Получаем токен из заголовков
    token = get_token_from_request(request)
    
    # Создаем клиент
    if token:
        user_yonote = YonoteClient(token=token)
    else:
        user_yonote = get_yonote_client()
        
    result = user_yonote.get_collections()
    
    if result.get("error"):
        raise HTTPException(status_code=500, detail=f"Yonote API error: {result['error']}")
    
    return result

@app.get("/yonote/api/test-connection")
async def test_yonote_connection():
    """Тестирует подключение к Yonote API"""
    result = yonote_client.get_collections()
    
    if result.get("error"):
        return {
            "success": False,
            "error": result.get("error"),
            "message": "Не удалось подключиться к Yonote API"
        }
    
    return {
        "success": True,
        "message": "Подключение к Yonote API установлено",
        "collections_count": len(result.get("data", [])) if isinstance(result, dict) else 0
    }


@app.get("/yonote/api/health")
@app.get("/whisper/api/health")
async def health_check():
    model_name = "base"
    return {
        "status": "ok",
        "model": model_name,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "version": "2.0",
        "features": ["transcription", "project_extraction", "yonote_integration", "video_support"]
    }


@app.get("/yonote", response_class=HTMLResponse)
async def yonote():
    """Страница для работы с Yonote"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="HTML file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


@app.get("/home", response_class=HTMLResponse)
async def home():
    """Домашняя страница со всеми сервисами"""
    try:
        with open("home.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="HTML file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


if __name__ == "__main__":
    logger.info("API Запущен с поддержкой Yonote и видео ----------")
    uvicorn.run(app, host="127.0.0.1", port=3534, reload=False)