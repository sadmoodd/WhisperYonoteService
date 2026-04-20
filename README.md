#  Whisper Yonote — AI Транскрибация аудио с интеграцией в Yonote

[![Laravel Version](https://img.shields.io/badge/Laravel-11.x-red.svg)](https://laravel.com)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)


## Скриншоты
<img width="1856" height="962" alt="homepage" src="https://github.com/user-attachments/assets/d5d2c4c9-0b5b-41e5-9132-a297cf274bb3" />
<img width="1856" height="962" alt="yonote-integration" src="https://github.com/user-attachments/assets/23c69e63-8fe5-48e1-9f9e-7bcf6740293e" />
<img width="1856" height="962" alt="clear-whisper-transcriber" src="https://github.com/user-attachments/assets/f1cbb0d3-6cb9-446b-b811-0ee37c08be52" />


##  О проекте

**Whisper Yonote** — это мощный сервис для автоматической транскрибации аудиофайлов с использованием AI-модели Whisper. Сервис предоставляет:

-  **Транскрибация аудио** в текст с высокой точностью
-  **Автоматическое реферирование** (генерация summary)
-  **Интеграция с Yonote** для автоматического создания задач

## Установка
```
git clone https://github.com/sadmoodd/WhisperYonoteService.git whisper-php-service
cd whisper-php-service
```
## Настройка и запуск
Запуск Laravel
```
cd backend
cp .env.example .env
composer install
php artisan key:generate

touch database/database.sqlite
chmod -R 775 database/database.sqlite

php artisan migrate
php artisan serve
```
Запуск Python
```
python3.10 -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt

python3.10 app.py
```

## Нюанс работы
Приложение работает в связке с ```https://github.com/sadmoodd/UniParserV2``` - преобразователем PDF файлов в Excel таблицы, поэтому для работы, помимо парсера, потребуется настроенный HTTP-сервер **nginx** с **ssl-сертификатом**. Пример конфига
```
server {
    listen _ ssl;
    server_name _;
    
    ssl_certificate /etc/nginx/ssl/selfsigned.crt;
    ssl_certificate_key /etc/nginx/ssl/selfsigned.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    client_max_body_size 100M;
    proxy_read_timeout 600;
    proxy_connect_timeout 600;
    proxy_send_timeout 600;

    # =========== WHISPER LARAVEL ===========
    location /whisper/ {
        proxy_pass http://127.0.0.1:8005/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # =========== WHISPER PYTHON API ===========    
    location /whisper/api/ {
        proxy_pass http://127.0.0.1:3534/whisper/api/;
        proxy_set_header Host $host;
    }

    # =========== PARSER LARAVEL  ===========
    location /parser/ {
        proxy_pass http://127.0.0.1:8001/;
        proxy_set_header Host $host;
    }

    # =========== PARSER PYTHON API  ===========
    location /parser/api/ {
        proxy_pass http://127.0.0.1:3637/api/;
        proxy_set_header Host $host;
    }

    # =========== YONOTE ===========
    location /yonote/ {
        proxy_pass http://127.0.0.1:3534/yonote;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
        
        # Важно для WebSocket и длинных запросов
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    location /yonote/api/ {
        proxy_pass http://127.0.0.1:3534/yonote/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
        
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # ========== HOMEPAGE ==========

    location /home/ {
        proxy_pass http://127.0.0.1:3534/home;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # =========== OLLAMA ===========
    location / {
        auth_basic "Ollama";
        auth_basic_user_file /etc/nginx/.ollama-passwd;
        proxy_pass http://127.0.0.1:11434/;
        proxy_set_header Host $host;
    }
}
```
Для каждого подсервиса потребуется служба **systemd**. Вместо Ollama можно использовать любую другую языковую модель, которая поддерживает интерфейс API.

