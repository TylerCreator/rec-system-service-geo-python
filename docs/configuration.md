# ⚙️ Конфигурация

Подробное описание настроек Service Recommendation System.

## Переменные окружения

Все настройки хранятся в файле `.env`. Используйте `env.example` как шаблон.

### База данных

```bash
# Хост базы данных
DB_HOST=localhost              # localhost для локального запуска
                              # postgresdb для Docker Compose

# Порт PostgreSQL
DB_PORT=5432                  # Стандартный порт PostgreSQL
DB_PORT=5431                  # Если используете Docker с маппингом портов

# Учетные данные
DB_USER=postgres              # Имя пользователя БД
DB_PASSWORD=postgres123       # Пароль (используйте сильный пароль в production!)
DB_NAME=rec_system           # Имя базы данных
```

**Важно для production:**
- `postgres123` - пароль для локальной разработки/тестирования
- В production используйте сложный пароль (минимум 32 символа)
- Не храните production пароли в git (`.env` должен быть в `.gitignore`)
- Используйте разные пароли для разных окружений

### Приложение

```bash
# Порт приложения
PORT=8080                     # Порт внутри контейнера

# Режим отладки
DEBUG=false                   # true - детальные логи, false - только важные
                             # В production всегда false!

# Автоматическое обновление
ENABLE_CRON=true              # true - включить cron задачи
                             # false - отключить автообновление
```

### Docker порты

```bash
# Внешние порты (для доступа с хоста)
NODE_LOCAL_PORT=6868          # Порт для доступа к API
POSTGRESDB_LOCAL_PORT=5431    # Порт для доступа к PostgreSQL

# Внутренние порты (внутри Docker)
NODE_DOCKER_PORT=8080         # Порт приложения в контейнере
POSTGRESDB_DOCKER_PORT=5432   # Стандартный порт PostgreSQL
```

**Маппинг портов:**
- `localhost:6868` → `app:8080` (приложение)
- `localhost:5431` → `postgresdb:5432` (база данных)

### SSL сертификаты

```bash
# Включение SSL
SSL_ENABLED=false             # true - использовать HTTPS
                             # false - использовать HTTP

# Пути к сертификатам (внутри контейнера)
SSL_KEY_PATH=/certs/privkey.pem
SSL_CERT_PATH=/certs/fullchain.pem
```

**Для включения SSL:**
1. Получите сертификаты (Let's Encrypt рекомендуется)
2. Смонтируйте их в контейнер через volumes
3. Установите `SSL_ENABLED=true`
4. Используйте `docker-compose-v2.yml`

### Внешние API

```bash
# URL CRIS API
CRIS_BASE_URL=http://cris.icc.ru

# Таймаут запросов (в секундах)
API_TIMEOUT=90                # Увеличьте для медленных подключений
```

### Пути к файлам

```bash
# CSV файл с экспортом вызовов
CSV_FILE_PATH=app/static/calls.csv

# JSON файл с рекомендациями
RECOMMENDATIONS_FILE_PATH=app/static/recomendations.json

# Python скрипт для обучения KNN
KNN_SCRIPT_PATH=app/static/knn.py
```

**Примечание:** Пути относительно корня проекта.

## Файл in_and_out_settings.json

Конфигурация отслеживания входов/выходов сервисов для построения композиций.

Расположение: `app/static/in_and_out_settings.json`

### Формат

```json
{
  "SERVICE_ID": {
    "input": {
      "parameter_name": "widget_type"
    },
    "output": {
      "parameter_name": "widget_type"
    }
  }
}
```

### Пример

```json
{
  "399": {
    "input": {
      "map": "edit",
      "new_layer_wms_link": "edit"
    },
    "output": {
      "map": "edit"
    }
  },
  "309": {
    "output": {
      "wms_link": "edit"
    }
  }
}
```

### Типы виджетов

- `edit` - отслеживается для связей между задачами
- `file` - файловый виджет
- `theme_select` - выбор датасета (игнорируется для связей задач)

### Добавление нового сервиса

1. Найдите ID сервиса в базе данных
2. Определите входные/выходные параметры
3. Добавьте запись в `in_and_out_settings.json`
4. Перезапустите приложение

## Cron расписание

При `ENABLE_CRON=true` автоматически выполняются:

### Обновление статистики

**Расписание:** Каждый час (0 минут)
**Действие:**
- Обновляет статистику сервисов
- Пересчитывает популярные сервисы
- Обновляет связи user-service

**Эквивалент:**
```bash
curl http://localhost:6868/update/statistics
```

### Обновление рекомендаций

**Расписание:** Каждые 6 часов (в 0:00, 6:00, 12:00, 18:00)
**Действие:**
- Экспортирует вызовы в CSV
- Запускает KNN обучение
- Генерирует рекомендации

**Эквивалент:**
```bash
curl http://localhost:6868/update/full
```

### Настройка расписания

Изменить расписание можно в `main.py`:

```python
# Статистика - каждый час
scheduler.add_job(
    update_statistics_cron,
    'cron',
    hour='*',      # Измените на нужное
    minute=0
)

# Рекомендации - каждые 6 часов
scheduler.add_job(
    update_all_cron,
    'cron',
    hour='*/6',    # Измените на нужное
    minute=0
)
```

## Настройки базы данных

### Размер connection pool

В `app/core/database.py`:

```python
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=10,        # Количество постоянных соединений
    max_overflow=20      # Дополнительные соединения при нагрузке
)
```

**Рекомендации:**
- Локальная разработка: `pool_size=5`, `max_overflow=5`
- Production малая нагрузка: `pool_size=10`, `max_overflow=20`
- Production высокая нагрузка: `pool_size=20`, `max_overflow=40`

### Настройки PostgreSQL

Для оптимизации производительности в `postgresql.conf`:

```ini
# Memory settings
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 16MB

# Connection settings
max_connections = 100

# WAL settings
wal_buffers = 16MB
checkpoint_completion_target = 0.9
```

## Логирование

### Уровни логирования

В `main.py`:

```python
import logging

# Уровень логирования
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

Уровни:
- `DEBUG` - все сообщения (только для разработки)
- `INFO` - информационные сообщения
- `WARNING` - предупреждения
- `ERROR` - ошибки
- `CRITICAL` - критические ошибки

### Логирование в файл

Добавьте в `main.py`:

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
```

### Ротация логов

Используйте `RotatingFileHandler`:

```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'server.log',
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=5            # Хранить 5 файлов
)
```

## CORS настройки

Для разрешения кросс-доменных запросов в `main.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # Frontend dev server
        "https://your-domain.com"     # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Для разрешения всех источников (не рекомендуется для production):**
```python
allow_origins=["*"]
```

## Переменные окружения для разных сред

### Development (.env.development)

```bash
DEBUG=true
ENABLE_CRON=false
DB_HOST=localhost
DB_PORT=5432
NODE_LOCAL_PORT=8080
CRIS_BASE_URL=http://cris.icc.ru
```

### Staging (.env.staging)

```bash
DEBUG=false
ENABLE_CRON=true
DB_HOST=postgresdb
DB_PORT=5432
NODE_LOCAL_PORT=6868
CRIS_BASE_URL=http://cris.icc.ru
SSL_ENABLED=true
```

### Production (.env.production)

```bash
# Application
DEBUG=false
ENABLE_CRON=true
NODE_LOCAL_PORT=6868
SSL_ENABLED=true

# Database
POSTGRESDB_USER=postgres
POSTGRESDB_ROOT_PASSWORD=<очень_сложный_пароль>  # !!! ОБЯЗАТЕЛЬНО ЗАМЕНИТЕ !!!
POSTGRESDB_DATABASE=rec_system

DB_HOST=postgresdb
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=<очень_сложный_пароль>  # Должен совпадать с POSTGRESDB_ROOT_PASSWORD
DB_NAME=rec_system

# External API
CRIS_BASE_URL=http://cris.icc.ru
```

### Использование

```bash
# Development
cp .env.development .env
docker-compose up

# Staging
cp .env.staging .env
docker-compose -f docker-compose-v2.yml up

# Production
cp .env.production .env
docker-compose -f docker-compose-v2.yml up -d
```

## Проверка конфигурации

### Просмотр текущей конфигурации

```bash
# В Docker
docker-compose exec app python -c "from app.core.config import settings; print(f'Port: {settings.PORT}\\nDebug: {settings.DEBUG}\\nCron: {settings.ENABLE_CRON}')"

# Локально
python -c "from app.core.config import settings; print(f'DB: {settings.DATABASE_URL}\\nPort: {settings.PORT}')"
```

### Валидация .env

```bash
# Проверка синтаксиса
docker-compose config

# Должно вывести корректный YAML без ошибок
```

## Примеры конфигураций

### Минимальная конфигурация (для локальной разработки)

```bash
# Database
POSTGRESDB_USER=postgres
POSTGRESDB_ROOT_PASSWORD=postgres123
POSTGRESDB_DATABASE=rec_system

# App
DB_HOST=postgresdb
DB_PASSWORD=postgres123
DB_NAME=rec_system
ENABLE_CRON=false
```

### Рекомендуемая конфигурация

```bash
# Database (для Docker Compose)
POSTGRESDB_USER=postgres
POSTGRESDB_ROOT_PASSWORD=postgres123  # Замените в production!
POSTGRESDB_DATABASE=rec_system

# Database Connection (для приложения)
DB_HOST=postgresdb
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=postgres123  # Должен совпадать с POSTGRESDB_ROOT_PASSWORD
DB_NAME=rec_system

# Application
PORT=8080
DEBUG=false
ENABLE_CRON=true

# External API
CRIS_BASE_URL=http://cris.icc.ru
API_TIMEOUT=90

# Files
CSV_FILE_PATH=app/static/calls.csv
RECOMMENDATIONS_FILE_PATH=app/static/recomendations.json
KNN_SCRIPT_PATH=app/static/knn.py
```

### Полная конфигурация

См. `env.example` в корне проекта.

