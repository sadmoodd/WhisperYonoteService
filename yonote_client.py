"""
Модуль для работы с API Yonote
"""

import os
import logging
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class YonoteClient:
    """Клиент для работы с API Yonote"""
    
    def __init__(self, base_url: str = None, token: str = None, default_collection_id: str = None):
        """
        Инициализация клиента Yonote
        
        Args:
            base_url: Базовый URL API Yonote (https://app.yonote.ru/api)
            token: API токен для аутентификации
            default_collection_id: ID коллекции по умолчанию
        """
        self.base_url = base_url or os.getenv("YONOTE_API_URL", "https://app.yonote.ru/api")
        self.token = token or os.getenv("YONOTE_TOKEN")
        self.default_collection_id = default_collection_id or os.getenv("YONOTE_COLLECTION_ID")
        
        if not self.token:
            logger.warning("YONOTE_TOKEN не задан в .env файле или параметрах")
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def _request(self, method: str, endpoint: str, data: dict = None, params: dict = None) -> Dict[str, Any]:
        """
        Выполняет запрос к API Yonote
        
        Args:
            method: HTTP метод (GET, POST, PUT, DELETE)
            endpoint: Эндпоинт API (без базового URL)
            data: Данные для POST/PUT запроса
            params: Параметры для GET запроса
        
        Returns:
            dict: Ответ API
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, headers=self.headers, timeout=30)
            elif method.upper() == "PUT":
                response = requests.put(url, json=data, headers=self.headers, timeout=30)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=self.headers, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Yonote API error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            return {"error": str(e), "success": False}
    
    def create_document(self, title: str, text: str, collection_id: str = None, 
                        publish: bool = True, parent_id: str = None) -> Dict[str, Any]:
        """
        Создает документ в Yonote
        
        Args:
            title: Заголовок документа
            text: Содержимое в формате Markdown
            collection_id: ID коллекции (если None, используется дефолтный)
            publish: Опубликовать сразу (True) или сохранить как черновик (False)
            parent_id: ID родительской страницы (для вложенности)
        
        Returns:
            dict: Ответ API с данными созданного документа
        """
        collection_id = collection_id or self.default_collection_id
        
        if not collection_id:
            return {"error": "collection_id не указан и не задан в .env", "success": False}
        
        payload = {
            "title": title,
            "collectionId": collection_id,
            "text": text,
            "publish": publish
        }
        
        if parent_id:
            payload["parentId"] = parent_id
        
        logger.info(f"Создаем документ в Yonote: {title}")
        return self._request("POST", "documents.create", payload)
    
    def update_document(self, document_id: str, title: str = None, text: str = None, 
                        publish: bool = None) -> Dict[str, Any]:
        """
        Обновляет существующий документ
        
        Args:
            document_id: ID документа
            title: Новый заголовок (опционально)
            text: Новое содержимое (опционально)
            publish: Изменить статус публикации (опционально)
        
        Returns:
            dict: Ответ API
        """
        payload = {}
        if title:
            payload["title"] = title
        if text:
            payload["text"] = text
        if publish is not None:
            payload["publish"] = publish
        
        if not payload:
            return {"error": "Нет данных для обновления", "success": False}
        
        return self._request("PUT", f"documents.update/{document_id}", payload)
    
    def get_document(self, document_id: str) -> Dict[str, Any]:
        """
        Получает информацию о документе
        
        Args:
            document_id: ID документа
        
        Returns:
            dict: Данные документа
        """
        return self._request("GET", f"documents.get/{document_id}")
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Удаляет документ
        
        Args:
            document_id: ID документа
        
        Returns:
            dict: Ответ API
        """
        return self._request("DELETE", f"documents.delete/{document_id}")
    
    def get_collections(self) -> Dict[str, Any]:
        """
        Получает список доступных коллекций
        
        Returns:
            dict: Список коллекций
        """
        return self._request("GET", "collections.list")
    
    def get_collection(self, collection_id: str) -> Dict[str, Any]:
        """
        Получает информацию о конкретной коллекции
        
        Args:
            collection_id: ID коллекции
        
        Returns:
            dict: Данные коллекции
        """
        return self._request("GET", f"collections.get/{collection_id}")
    
    def search_documents(self, query: str, collection_id: str = None, limit: int = 10) -> Dict[str, Any]:
        """
        Поиск документов
        
        Args:
            query: Поисковый запрос
            collection_id: ID коллекции (опционально)
            limit: Максимальное количество результатов
        
        Returns:
            dict: Результаты поиска
        """
        params = {"q": query, "limit": limit}
        if collection_id:
            params["collectionId"] = collection_id
        return self._request("GET", "documents.search", params)
    
    def create_comment(self, document_id: str, text: str) -> Dict[str, Any]:
        """
        Добавляет комментарий к документу
        
        Args:
            document_id: ID документа
            text: Текст комментария
        
        Returns:
            dict: Ответ API
        """
        payload = {
            "documentId": document_id,
            "text": text
        }
        return self._request("POST", "comments.create", payload)
    
    def get_comments(self, document_id: str) -> Dict[str, Any]:
        """
        Получает комментарии к документу
        
        Args:
            document_id: ID документа
        
        Returns:
            dict: Список комментариев
        """
        return self._request("GET", f"comments.list/{document_id}")


def format_project_to_markdown(project_data: dict) -> str:
    """
    Форматирует данные проекта в Markdown для Yonote
    
    Args:
        project_data: Словарь с данными проекта (из extract_project_and_tasks_from_transcription)
    
    Returns:
        str: Текст в формате Markdown
    """
    project = project_data.get("project", {})
    tasks = project_data.get("tasks", [])
    context = project_data.get("context", {})
    
    # Заголовок
    md = f"# {project.get('name', 'Новый проект')}\n\n"
    
    # Описание
    if project.get('description'):
        md += f"{project.get('description')}\n\n"
    
    # Метрики
    if project.get('key_metrics'):
        md += f"**📊 Ключевые метрики:** {project.get('key_metrics')}\n\n"
    
    # Задачи
    md += "## 📋 Задачи\n\n"
    
    # Статистика задач
    total_tasks = len(tasks)
    if total_tasks > 0:
        md += f"**Всего задач:** {total_tasks}\n\n"
    
    for idx, task in enumerate(tasks, 1):
        priority_emoji = {
            "high": "🔴",
            "medium": "🟡",
            "low": "🟢"
        }.get(task.get("priority", "medium"), "⚪")
        
        md += f"### {idx}. {priority_emoji} {task.get('title', 'Задача')}\n\n"
        
        if task.get('description'):
            md += f"{task.get('description')}\n\n"
        
        # Метаданные задачи
        meta_parts = []
        if task.get('assignee'):
            meta_parts.append(f"**Исполнитель:** {task.get('assignee')}")
        if task.get('deadline'):
            meta_parts.append(f"**Дедлайн:** {task.get('deadline')}")
        if task.get('estimated_hours'):
            meta_parts.append(f"**Оценка:** {task.get('estimated_hours')} ч.")
        
        if meta_parts:
            md += " | ".join(meta_parts) + "\n\n"
    
    # Контекст
    if context.get('decisions') or context.get('risks') or context.get('dependencies'):
        md += "## 📌 Контекст\n\n"
        
        if context.get('decisions'):
            md += "### Решения\n\n"
            for decision in context['decisions']:
                md += f"- {decision}\n"
            md += "\n"
        
        if context.get('dependencies'):
            md += "### Зависимости\n\n"
            for dep in context['dependencies']:
                md += f"- {dep}\n"
            md += "\n"
        
        if context.get('risks'):
            md += "### ⚠️ Риски и блокеры\n\n"
            for risk in context['risks']:
                md += f"- {risk}\n"
            md += "\n"
    
    if context.get('important_notes'):
        md += f"**💡 Важно:** {context.get('important_notes')}\n\n"
    
    # Футер с датой
    md += f"\n---\n*Создано автоматически из аудио {datetime.now().strftime('%d.%m.%Y %H:%M')}*"
    
    return md


# Создаем глобальный экземпляр клиента
def get_yonote_client() -> YonoteClient:
    """Возвращает экземпляр клиента Yonote с настройками из .env"""
    return YonoteClient()