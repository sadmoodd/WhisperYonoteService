# ========== НОВЫЕ ФУНКЦИИ ДЛЯ РАСШИРЕННОГО ФУНКЦИОНАЛА ==========

def parse_voice_command_for_intent(transcription: str) -> dict:
    """
    Определяет намерение пользователя из голосовой команды
    Возвращает: {
        "action": "find_project" | "find_task" | "add_tasks" | "create_project" | "update_project",
        "target_name": str,  # название проекта или задачи
        "project_name": str,  # название проекта (для add_tasks)
        "tasks": list,  # список задач (для add_tasks)
        "query": str  # поисковый запрос
    }
    """
    prompt = f"""
Проанализируй голосовую команду и определи намерение пользователя.

Возможные намерения:
1. FIND_PROJECT - найти проект по названию
2. FIND_TASK - найти задачу по названию
3. ADD_TASKS - добавить новые задачи в существующий проект
4. CREATE_PROJECT - создать новый проект с задачами
5. UPDATE_PROJECT - обновить существующий проект

Команда: "{transcription}"

Верни JSON:
{{
    "action": "FIND_PROJECT|FIND_TASK|ADD_TASKS|CREATE_PROJECT|UPDATE_PROJECT",
    "target_name": "название проекта или задачи для поиска",
    "project_name": "название проекта (для ADD_TASKS)",
    "tasks": ["задача 1", "задача 2"],
    "query": "поисковый запрос"
}}
"""
    
    payload = {
        "model": os.getenv("MODEL_NAME", 'gemma3:1b'),
        "messages": [
            {"role": "system", "content": "Ты анализатор команд. Отвечай только JSON."},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "temperature": 0.3
    }
    
    try:
        response_text = query(payload)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        logger.error(f"Intent parsing error: {e}")
    
    return {"action": "CREATE_PROJECT", "target_name": "", "project_name": "", "tasks": [], "query": transcription}


def find_task_in_project(yonote_client, document_id: str, task_query: str) -> list:
    """
    Ищет задачи внутри проекта по названию
    """
    try:
        doc = yonote_client.get_document(document_id)
        if doc.get("error"):
            return []
        
        content = doc.get("text", "")
        project_data = extract_project_and_tasks_from_transcription(content[:5000])
        
        found_tasks = []
        for task in project_data.get("tasks", []):
            task_title = task.get("title", "").lower()
            if task_query.lower() in task_title:
                found_tasks.append({
                    "title": task.get("title"),
                    "description": task.get("description"),
                    "assignee": task.get("assignee"),
                    "priority": task.get("priority"),
                    "deadline": task.get("deadline"),
                    "status": task.get("status", "open")
                })
        
        return found_tasks
    except Exception as e:
        logger.error(f"Task search error: {e}")
        return []


def search_all_projects(yonote_client, query: str, collection_id: str = None) -> list:
    """
    Ищет проекты по названию и возвращает список
    """
    try:
        result = yonote_client.search_documents(query, collection_id)
        
        projects = []
        if result.get("data"):
            for doc in result["data"]:
                projects.append({
                    "id": doc.get("id"),
                    "title": doc.get("title"),
                    "url": doc.get("url"),
                    "updated_at": doc.get("updatedAt"),
                    "collection_id": doc.get("collectionId")
                })
        return projects
    except Exception as e:
        logger.error(f"Project search error: {e}")
        return []


def add_tasks_to_project(yonote_client, document_id: str, new_tasks: list) -> dict:
    """
    Добавляет новые задачи в существующий проект
    """
    try:
        # Получаем текущий документ
        current_doc = yonote_client.get_document(document_id)
        if current_doc.get("error"):
            return {"error": current_doc["error"]}
        
        content = current_doc.get("text", "")
        
        # Извлекаем текущую структуру проекта
        project_data = extract_project_and_tasks_from_transcription(content[:5000])
        
        # Добавляем новые задачи
        for task_title in new_tasks:
            project_data["tasks"].append({
                "title": task_title,
                "description": "",
                "assignee": None,
                "priority": "medium",
                "deadline": None,
                "estimated_hours": None,
                "status": "open"
            })
        
        # Форматируем и обновляем
        new_markdown = format_project_to_markdown(project_data)
        update_result = yonote_client.update_document(
            document_id=document_id,
            text=new_markdown,
            publish=True
        )
        
        return update_result
    except Exception as e:
        logger.error(f"Add tasks error: {e}")
        return {"error": str(e)}


# ========== НОВЫЙ УМНЫЙ ЭНДПОИНТ ДЛЯ ГОЛОСОВЫХ КОМАНД ==========

@app.post("/yonote/api/voice-command")
async def voice_command(
    request: Request,
    file: UploadFile = File(...),
    collection_id: Optional[str] = Form(None),
    auto_execute: bool = Form(True)
):
    """
    Универсальный эндпоинт для обработки голосовых команд:
    - "найди проект Маркетинг"
    - "найди задачу дизайн"
    - "запиши задачи в проект ККК: задача 1, задача 2"
    - "создай проект ХХ с задачами..."
    """
    token = get_token_from_request(request)
    
    if token:
        yonote_client = YonoteClient(token=token)
    else:
        yonote_client = default_yonote_client if 'default_yonote_client' in globals() else get_yonote_client()
    
    # Транскрибируем аудио
    upload_file = file
    if not upload_file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Сохраняем и транскрибируем (используем существующую логику)
    with tempfile.TemporaryDirectory() as tmpdir:
        file_ext = upload_file.filename.split('.')[-1].lower()
        filepath = os.path.join(tmpdir, f"audio.{file_ext}")
        
        content = await upload_file.read()
        with open(filepath, "wb") as f:
            f.write(content)
        
        # Транскрипция
        audio_segment = AudioSegment.from_file(filepath)
        chunks = [audio_segment[i:i + 60000] for i in range(0, len(audio_segment), 60000)]
        
        full_transcription = ""
        for chunk in chunks:
            chunk_path = os.path.join(tmpdir, "chunk.wav")
            chunk.export(chunk_path, format="wav")
            result = await transcribe_chunk_async(chunk_path, 0)
            full_transcription += " " + result["text"]
        
        full_transcription = full_transcription.strip()
    
    # Анализируем намерение
    intent = parse_voice_command_for_intent(full_transcription)
    action = intent.get("action", "CREATE_PROJECT")
    
    response = {
        "success": True,
        "transcription": full_transcription,
        "intent": intent,
        "action": action
    }
    
    # ========== ВЫПОЛНЯЕМ ДЕЙСТВИЕ ==========
    
    if action == "FIND_PROJECT":
        query = intent.get("target_name") or intent.get("query") or full_transcription
        projects = search_all_projects(yonote_client, query, collection_id)
        response["result"] = {
            "found": len(projects),
            "projects": projects
        }
        response["message"] = f"Найдено проектов: {len(projects)}"
    
    elif action == "FIND_TASK":
        query = intent.get("target_name") or intent.get("query") or full_transcription
        
        # Сначала ищем проекты
        projects = search_all_projects(yonote_client, "", collection_id)
        
        all_found_tasks = []
        for project in projects[:5]:  # Проверяем первые 5 проектов
            tasks = find_task_in_project(yonote_client, project["id"], query)
            for task in tasks:
                task["project_name"] = project["title"]
                task["project_id"] = project["id"]
                all_found_tasks.append(task)
        
        response["result"] = {
            "found": len(all_found_tasks),
            "tasks": all_found_tasks
        }
        response["message"] = f"Найдено задач: {len(all_found_tasks)}"
    
    elif action == "ADD_TASKS":
        project_name = intent.get("project_name", "")
        tasks_to_add = intent.get("tasks", [])
        
        if not tasks_to_add:
            # Пробуем извлечь задачи из транскрипции напрямую
            project_data = extract_project_and_tasks_from_transcription(full_transcription)
            tasks_to_add = [t.get("title") for t in project_data.get("tasks", [])]
            project_name = project_data.get("project", {}).get("name", project_name)
        
        # Ищем проект
        document_id = find_project_by_name_with_client(yonote_client, project_name, collection_id)
        
        if document_id and auto_execute:
            result = add_tasks_to_project(yonote_client, document_id, tasks_to_add)
            response["result"] = {
                "project_found": True,
                "document_id": document_id,
                "tasks_added": tasks_to_add,
                "update_result": result
            }
            response["message"] = f"Добавлено {len(tasks_to_add)} задач в проект '{project_name}'"
        else:
            response["result"] = {
                "project_found": document_id is not None,
                "project_name": project_name,
                "tasks_to_add": tasks_to_add
            }
            response["message"] = f"Проект '{project_name}' {'найден' if document_id else 'не найден'}"
    
    elif action in ["CREATE_PROJECT", "UPDATE_PROJECT"]:
        # Используем существующую логику
        project_data = extract_project_and_tasks_from_transcription(full_transcription)
        
        if auto_execute:
            markdown = format_project_to_markdown(project_data)
            title = project_data.get("project", {}).get("name", "Новый проект")
            
            if action == "UPDATE_PROJECT":
                project_name = intent.get("project_name") or title
                document_id = find_project_by_name_with_client(yonote_client, project_name, collection_id)
                if document_id:
                    result = yonote_client.update_document(document_id, markdown, publish=True)
                    response["result"] = {"updated": True, "document_id": document_id}
                else:
                    result = yonote_client.create_document(title, markdown, collection_id, publish=True)
                    response["result"] = {"created": True, "document_id": result.get("id")}
            else:
                result = yonote_client.create_document(title, markdown, collection_id, publish=True)
                response["result"] = {"created": True, "document_id": result.get("id")}
            
            response["project_data"] = project_data
        else:
            response["project_data"] = project_data
            response["message"] = "Проект подготовлен (auto_execute=False)"
    
    return response


# Вспомогательная функция
def find_project_by_name_with_client(client: YonoteClient, project_name: str, collection_id: str = None) -> Optional[str]:
    """Ищет проект используя переданный клиент"""
    if not project_name:
        return None
    try:
        result = client.search_documents(project_name, collection_id)
        if result.get("data") and len(result["data"]) > 0:
            for doc in result["data"]:
                if project_name.lower() in doc.get("title", "").lower():
                    return doc.get("id")
            return result["data"][0].get("id")
        return None
    except Exception as e:
        logger.error(f"Find project error: {e}")
        return None


# ========== ДОПОЛНИТЕЛЬНЫЕ УДОБНЫЕ ЭНДПОИНТЫ ==========

@app.get("/yonote/api/search")
async def search(
    request: Request,
    q: str,
    type: str = "all",  # all, projects, tasks
    collection_id: Optional[str] = None
):
    """Универсальный поиск по проектам и задачам"""
    token = get_token_from_request(request)
    client = YonoteClient(token=token) if token else get_yonote_client()
    
    results = {"projects": [], "tasks": []}
    
    if type in ["all", "projects"]:
        results["projects"] = search_all_projects(client, q, collection_id)
    
    if type in ["all", "tasks"]:
        projects = results["projects"] if results["projects"] else search_all_projects(client, "", collection_id)
        for project in projects[:10]:
            tasks = find_task_in_project(client, project["id"], q)
            for task in tasks:
                task["project_name"] = project["title"]
                task["project_id"] = project["id"]
                results["tasks"].append(task)
    
    return {
        "success": True,
        "query": q,
        "type": type,
        "results": results,
        "total": {
            "projects": len(results["projects"]),
            "tasks": len(results["tasks"])
        }
    }


@app.post("/yonote/api/project/{document_id}/add-tasks")
async def add_tasks_to_project_endpoint(
    request: Request,
    document_id: str,
    tasks: List[str] = None
):
    """Добавляет задачи в существующий проект"""
    if not tasks:
        data = await request.json()
        tasks = data.get("tasks", [])
    
    if not tasks:
        raise HTTPException(status_code=400, detail="No tasks provided")
    
    token = get_token_from_request(request)
    client = YonoteClient(api_key=token) if token else get_yonote_client()
    
    result = add_tasks_to_project(client, document_id, tasks)
    
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    
    return {
        "success": True,
        "document_id": document_id,
        "tasks_added": len(tasks),
        "tasks": tasks
    }