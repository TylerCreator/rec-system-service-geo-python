# Service Recommendation System

> Система рекомендаций сервисов на основе анализа истории вызовов и композиций workflows

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)](https://www.postgresql.org/)

## 📋 Описание

API-сервис для системы рекомендаций геопространственных сервисов. Система:

- 🔍 **Анализирует** историю вызовов сервисов пользователями
- 🔗 **Восстанавливает** композиции сервисов (workflows) из истории
- 🤖 **Рекомендует** сервисы на основе ML (KNN, popularity-based + расширяемо)
- ⚡ **Быстрые рекомендации** с кэшированием (10-50ms против 2-5s)
- 📊 **Собирает** статистику использования
- 🔄 **Автоматически обновляет** данные по расписанию (cron)

## 🚀 Быстрый старт

### Запуск с Docker (рекомендуется)

```bash
# 1. Скопировать настройки
cp env.example .env

# 2. Запустить сервисы
docker-compose up -d --build

# 3. Открыть документацию API
open http://localhost:6868/docs
```

### Запуск без Docker

```bash
# 1. Установить зависимости
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Настроить окружение
cp env.example .env
# Отредактировать .env для подключения к PostgreSQL

# 3. Запустить приложение
python main.py
```

## 🎯 Эндпоинты и интерфейс (актуально по коду)

После запуска доступны:

- **📚 API Docs (Swagger):** http://localhost:6868/docs
- **📖 ReDoc:** http://localhost:6868/redoc
- **❤️ Health Check:** http://localhost:6868/

### Root и admin

| Method | Path | Интерфейс | Ответ |
| --- | --- | --- | --- |
| GET | `/` | Без параметров | `{ "message", "status", "version", "timestamp" }` |
| GET | `/admin/run-cron` | Без параметров | `{ "message", "result", "timestamp" }` |

### Calls

| Method | Path | Интерфейс | Ответ |
| --- | --- | --- | --- |
| GET | `/calls/` | Без параметров | `Call[]` (история вызовов, сортировка по `id desc`) |
| GET | `/calls/incr` | Без параметров | `{ "data": [...] }` |
| GET | `/calls/update-calls` | Без параметров | `{ "message": "Calls updated successfully" }` |
| GET | `/calls/dump-csv` | Без параметров | `{ "message": "CSV file created successfully", "status": 200 }` |

`Call` содержит поля: `id`, `classname`, `console_output`, `created_by`, `created_on`, `edited_by`, `edited_on`, `end_time`, `error_output`, `input`, `input_data`, `input_params`, `is_deleted`, `mid`, `os_pid`, `owner`, `result`, `start_time`, `status`.

### Services

| Method | Path | Интерфейс | Ответ |
| --- | --- | --- | --- |
| GET | `/services/` | Query: `user?: string`, `limit?: int` | `Service[]` |
| GET | `/services/legacy/getRecomendations` | Query: `user_id: string` | `int[]` (deprecated) |
| GET | `/services/legacy/getRecomendation` | Query: `user_id?: string` | `int[]` (deprecated) |
| GET | `/services/recommendations/{user_id}` | Path: `user_id: string`; Query: `n=10 (1..100)`, `algorithm?: string`, `period?: string`, `min_calls?: int>=1`, `ids_only=false` | `RecommendationResult` или `int[]` при `ids_only=true` |
| POST | `/services/recommendations/batch` | JSON body: `{ "user_ids": string[], "n"?: int(1..100), "algorithm"?: string, "ids_only"?: bool }` | `{ "results": {...}, "total_users", "algorithm_used" }` или `{ "userId": int[] }` |
| GET | `/services/recommendations/algorithms` | Query: `algorithm?: string` | `{ "<algorithm>": {...} }` или `{ "error": ... }` |
| GET | `/services/recommendations/stats` | Без параметров | `{ "is_initialized", "algorithms", "default_algorithm", "data_loader", "cache" }` |
| POST | `/services/recommendations/refresh` | Без body | `{ "success": bool, "message": string, "stats"?: {...} }` |
| GET | `/services/popular` | Query: `type="any"`, `limit=20`, `period="all"`, `min_calls=1`, `user_id?: string`, `ids_only=false` | `PopularResponse` или `int[]` при `ids_only=true` |
| GET | `/services/parameters/{service_id}` | Path: `service_id: int`; Query: `user?: string`, `limit=100`, `unique="true"` | `ServiceParametersResponse` или `{ "error": "Service not found", "serviceId": ... }` |

`Service` содержит поля: `id`, `name`, `subject`, `type`, `description`, `number_of_calls`, `actionview`, `actionmodify`, `map_reduce_specification`, `params`, `js_body`, `wpsservers`, `wpsmethod`, `status`, `output_params`, `wms_link`, `wms_layer_name`, `is_deleted`, `created_by`, `edited_by`, `edited_on`, `created_on`, `classname`.

`RecommendationResult`:
```json
{
  "user_id": "user123",
  "recommendations": [
    {
      "service_id": 399,
      "score": 0.91,
      "algorithm": "knn",
      "confidence": 0.87,
      "reason": "similar users",
      "metadata": {}
    }
  ],
  "algorithm_used": "knn",
  "fallback_used": false,
  "execution_time_ms": 12.3,
  "timestamp": "2026-05-22T12:00:00",
  "count": 1,
  "metadata": {
    "cache_hit": false,
    "requested_algorithm": "knn",
    "n_requested": 10,
    "n_returned": 1
  }
}
```

`PopularResponse`:
```json
{
  "items": [
    {
      "itemId": 1002120,
      "originalId": 2120,
      "itemName": "Dataset GUID",
      "itemType": "dataset",
      "serviceType": "table",
      "itemDescription": "Dataset with GUID: ...",
      "itemSubject": "",
      "callCount": 42,
      "uniqueUsers": 10,
      "popularity": 4.2,
      "rank": 1,
      "serviceId": 1002120,
      "serviceName": "Dataset GUID"
    }
  ],
  "services": [],
  "meta": {
    "total_services_in_db": 120,
    "total_datasets_in_db": 500,
    "total_items_in_db": 620,
    "total_successful_calls": 1000,
    "filtered_by_type": "any",
    "time_period": "all",
    "filtered_by_user": null,
    "user_specific": false,
    "min_calls_threshold": 1,
    "limit": 20,
    "returned_count": 20,
    "breakdown": {
      "services": 12,
      "datasets": 8
    },
    "time_filter_applied": false,
    "generated_at": "2026-05-22T12:00:00"
  }
}
```

`ServiceParametersResponse`:
```json
{
  "service": {
    "id": 399,
    "name": "mapcombine",
    "description": "...",
    "type": "service"
  },
  "parameters": [
    {
      "callId": 1,
      "owner": "user123",
      "timestamp": "2026-05-22T12:00:00",
      "status": "TASK_SUCCEEDED",
      "parameters": {}
    }
  ],
  "analysis": {},
  "setsAnalysis": {},
  "schema": {},
  "totalCalls": 10,
  "returnedParameters": 5,
  "filters": {
    "user": null,
    "limit": 100,
    "unique": true
  }
}
```

### Datasets

| Method | Path | Интерфейс | Ответ |
| --- | --- | --- | --- |
| GET | `/datasets/update` | Без параметров | `{ "message": "Datasets updated successfully" }` |

### Compositions

| Method | Path | Интерфейс | Ответ |
| --- | --- | --- | --- |
| GET | `/compositions/recover` | Без параметров | `{ "success", "message", "compositionsCount", "usersCount" }` |
| GET | `/compositions/recoverNew` | Без параметров | `{ "success", "message", "compositionsCount", "serviceSequencesCount", "servicesCount", "datasetsCount" }` |
| GET | `/compositions/` | Без параметров | `{ "id", "nodes", "links" }[]` |
| GET | `/compositions/stats` | Без параметров | `{ "totalCompositions", "totalNodes", "avgNodesPerComposition", "topServices" }` |

### Update

| Method | Path | Интерфейс | Ответ |
| --- | --- | --- | --- |
| GET | `/update/all` | Без параметров | `{ "message", "results": string[], "timestamp" }` |
| GET | `/update/recomendations` | Без параметров | `{ "success", "message"/"error", "timestamp", ... }` |
| GET | `/update/statistic` | Без параметров | `{ "success", "message", "result", "timestamp" }` |
| GET | `/update/full` | Без параметров | `{ "success", "message", "executionTime", "results", "hasErrors", "timestamp" }` |
| GET | `/update/local` | Без параметров | `{ "message", "result", "timestamp" }` |

### Sequential recommendations

| Method | Path | Интерфейс | Ответ |
| --- | --- | --- | --- |
| POST | `/sequential/predict` | JSON body: `{ "sequence": int[], "n"?: int(1..20), "ids_only"?: bool }` | `{ "sequence", "next_services", "count", "algorithm" }` или `int[]` |
| POST | `/sequential/possible` | JSON body: `{ "sequence": int[] }` | `{ "sequence", "possible_next_services", "count", "source" }` |
| POST | `/sequential/train` | Без body | `{ "success", "message", "model_info" }` |
| GET | `/sequential/info` | Без параметров | `model_info` (словарь параметров/статуса модели) |
| GET | `/sequential/health` | Без параметров | `{ "status", "model_loaded", "dag_available", "nodes_count" }` |
| POST | `/sequential/tables/predict` | JSON body: `{ "table_sequence": int[], "n"?: int(1..20), "ids_only"?: bool }` | `{ "table_sequence", "next_tables", "count", "algorithm", "type" }` или `int[]` |
| POST | `/sequential/tables/possible` | JSON body: `{ "table_sequence": int[] }` | `{ "table_sequence", "possible_next_tables", "count", "source", "type" }` |

### Примечания

- Для неизвестных путей API возвращает `404` с телом: `{ "message": "страница не найдена" }`.
- ⚠️ Legacy-маршруты `/services/legacy/getRecomendations` и `/services/legacy/getRecomendation` оставлены для обратной совместимости.

## 📁 Структура проекта

```
rec-system-services-geo-python/
├── main.py                      # Точка входа приложения
├── app/
│   ├── core/                    # Конфигурация и БД
│   │   ├── config.py
│   │   └── database.py
│   ├── models/                  # SQLAlchemy модели
│   │   └── models.py
│   ├── routers/                 # API эндпоинты
│   │   ├── calls.py
│   │   ├── services.py
│   │   ├── compositions.py
│   │   ├── update.py
│   │   └── sequential.py       # 🆕 Sequential recommendations
│   ├── services/                # Бизнес-логика
│   │   ├── calls_service.py
│   │   ├── compositions_service.py
│   │   ├── recommendations_service.py  # 🆕 Новый сервис
│   │   ├── sequential_recommendations_service.py  # 🆕 Sequential сервис
│   │   ├── recommendations/     # 🆕 Модульная архитектура ML
│   │   │   ├── engine.py       # Движок рекомендаций
│   │   │   ├── data_loader.py  # Загрузка данных с кэшем
│   │   │   ├── cache.py        # LRU кэш
│   │   │   ├── algorithms/     # Алгоритмы рекомендаций
│   │   │   │   ├── knn.py                  # KNN collaborative filtering
│   │   │   │   ├── popularity.py           # Popularity-based
│   │   │   │   ├── analytics_popularity.py # Real-time analytics
│   │   │   │   └── sequential_dagnn.py     # 🆕 DAGNN для workflow
│   │   │   └── models/         # Data models
│   │   │       ├── recommendation.py
│   │   │       └── user_profile.py
│   │   └── update_service.py
│   └── static/                  # Статические файлы
│       ├── calls.csv
│       ├── compositionsDAG.json
│       ├── recomendations.json
│       ├── knn.py              # ML модель (legacy)
│       └── in_and_out_settings.json
├── requirements.txt             # Python зависимости
├── Dockerfile                   # Docker образ
├── docker-compose.yml          # Docker Compose
└── docs/                       # Документация
    ├── deployment.md
    ├── configuration.md
    └── migration.md
```

## ⚙️ Конфигурация

Основные переменные окружения в `.env`:

```bash
# База данных (для Docker Compose)
POSTGRESDB_USER=postgres
POSTGRESDB_ROOT_PASSWORD=root          # Для dev/test; замените в production!
POSTGRESDB_DATABASE=compositions
POSTGRESDB_LOCAL_PORT=5431

# База данных (подключение приложения)
DB_HOST=localhost                      # localhost для локального запуска
                                       # postgresdb для Docker Compose
DB_PORT=5431                          # 5431 для локального доступа к Docker
                                       # 5432 для локального PostgreSQL
DB_USER=postgres
DB_PASSWORD=root                       # Должен совпадать с POSTGRESDB_ROOT_PASSWORD
DB_NAME=compositions

# Приложение
PORT=8080
DEBUG=false
ENABLE_CRON=true

# Внешние API
CRIS_BASE_URL=http://cris.icc.ru
API_TIMEOUT=90

# Файлы
CSV_FILE_PATH=app/static/calls.csv
RECOMMENDATIONS_FILE_PATH=app/static/recomendations.json
KNN_SCRIPT_PATH=app/static/knn.py
```

> **⚠️ Важно для production:** Замените `root` на сильный пароль (минимум 32 символа) в обоих местах: `POSTGRESDB_ROOT_PASSWORD` и `DB_PASSWORD`!

Подробнее: [docs/configuration.md](docs/configuration.md)

## 🐳 Docker

### Локальная разработка

```bash
# Запуск с автоперезагрузкой
docker-compose up --build

# Остановка
docker-compose down

# Просмотр логов
docker-compose logs -f app
```

### Production развертывание

```bash
# С SSL сертификатами
docker-compose -f docker-compose-v2.yml up -d --build

# Проверка статуса
docker-compose -f docker-compose-v2.yml ps
```

Подробнее: [docs/deployment.md](docs/deployment.md)

## 🔄 Обновление данных

Система автоматически обновляет данные по расписанию:

- **Статистика:** каждый час
- **Рекомендации:** каждые 6 часов
- **Композиции:** при необходимости

Ручное обновление:

```bash
# Полное обновление всех данных
curl http://localhost:6868/update/full

# Только статистика
curl http://localhost:6868/update/statistic

# Восстановление композиций
curl http://localhost:6868/compositions/recover
```

## 🤖 Машинное обучение

### Новая архитектура рекомендаций (v2)

Система поддерживает **множественные алгоритмы** с автоматическим выбором:

#### 1. **KNN (Collaborative Filtering)**
- Анализирует матрицу пользователь-сервис
- Находит похожих пользователей (cosine similarity)
- Рекомендует популярные сервисы среди соседей
- **Используется для:** опытных пользователей (≥3 вызовов)

#### 2. **Popularity-based**
- Глобально популярные сервисы
- Быстрый и простой алгоритм
- **Используется для:** новых пользователей (cold start)

#### 3. **Sequential DAGNN (Graph Neural Network)**
- Предсказывает следующий сервис/таблицу в workflow
- Основан на DAG структуре композиций
- Использует Graph Neural Networks (DAGNN)
- **Два режима:**
  - Services: предсказание следующего сервиса
  - Tables: предсказание следующей таблицы (игнорирует промежуточные сервисы)
- **Используется для:** продолжения существующих последовательностей

#### Ключевые преимущества:
- ⚡ **Быстро:** 10-50ms вместо 2-5 секунд
- 💾 **Кэш:** LRU cache с TTL для мгновенных повторных запросов
- 🎯 **Умный выбор:** автоматический fallback для новых пользователей
- 🔧 **Расширяемо:** легко добавлять новые алгоритмы
- 📊 **Метрики:** встроенный мониторинг и статистика

#### Примеры использования:

```bash
# Автоматический выбор алгоритма (рекомендуется)
curl "http://localhost:8080/services/recommendations/user123"

# Явный выбор алгоритма
curl "http://localhost:8080/services/recommendations/user123?algorithm=knn"

# Последовательные рекомендации (следующий сервис в workflow)
curl -X POST "http://localhost:8080/sequential/predict" \
  -H "Content-Type: application/json" \
  -d '{"sequence": [123, 456, 789], "n": 5}'

# Последовательные рекомендации таблиц (следующая таблица в workflow)
curl -X POST "http://localhost:8080/sequential/tables/predict" \
  -H "Content-Type: application/json" \
  -d '{"table_sequence": [1002120, 1001211], "n": 5, "ids_only": true}'
```

#### Инициализация:
Модели автоматически инициализируются при старте приложения:
```bash
python main.py
# ✅ Recommendation engine initialized
# ✅ Sequential DAGNN engine initialized
```

Обновление моделей:
```bash
# Обновить user-based модели (KNN, Popularity)
curl -X POST "http://localhost:8080/services/recommendations/refresh"

# Обучить Sequential DAGNN (первый раз или после изменений композиций)
curl -X POST "http://localhost:8080/sequential/train"

# Полное обновление системы (все модели + данные)
curl "http://localhost:8080/update/full"
```

## 📊 Композиции сервисов

Система восстанавливает workflows из истории вызовов:

- Анализирует связи между вызовами через входные/выходные данные
- Строит граф композиций (DAG)
- Сохраняет в `app/static/compositionsDAG.json`

Особенности:
- Поддержка WMS-сервисов
- Поддержка сервиса mapcombine (ID: 399)
- Отслеживание промежуточных результатов через `edit` виджеты

## 🧪 Тестирование

```bash
# Проверка здоровья
curl http://localhost:6868/

# Статистика движка рекомендаций
curl http://localhost:6868/services/recommendations/stats

# Рекомендации для пользователя
curl http://localhost:6868/services/recommendations/50f7a1d80d58140037000006
```

## 📚 Документация

### Основная
- **[Быстрый старт](docs/quickstart.md)** - детальная инструкция по запуску
- **[Конфигурация](docs/configuration.md)** - настройка приложения
- **[Развертывание](docs/deployment.md)** - деплой на сервер
- **[API](docs/api.md)** - описание эндпоинтов
- **[Миграция](docs/migration.md)** - переход с Node.js версии

### 🆕 Система рекомендаций
- **[Архитектура](docs/recommendations_architecture.md)** - дизайн новой системы
- **[Использование](docs/recommendations_usage.md)** - примеры и best practices
- **[Миграция на v2](docs/recommendations_migration.md)** - переход на новый API

## 🛠 Технологии

- **[FastAPI](https://fastapi.tiangolo.com/)** - современный веб-фреймворк
- **[SQLAlchemy](https://www.sqlalchemy.org/)** - ORM для работы с БД
- **[PostgreSQL](https://www.postgresql.org/)** - реляционная база данных
- **[Scikit-learn](https://scikit-learn.org/)** - машинное обучение (KNN, классификация)
- **[PyTorch](https://pytorch.org/)** - deep learning framework
- **[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)** - Graph Neural Networks
- **[NetworkX](https://networkx.org/)** - анализ графов
- **[APScheduler](https://apscheduler.readthedocs.io/)** - планировщик задач
- **[Docker](https://www.docker.com/)** - контейнеризация

## 🔧 Разработка

### Требования

- Python 3.9+
- PostgreSQL 13+
- PyTorch 2.1+ (для Sequential DAGNN)
- CUDA (опционально, для GPU ускорения)
- Docker & Docker Compose (опционально)

### Структура базы данных

Основные таблицы:
- `Calls` - история вызовов сервисов
- `Services` - метаданные сервисов
- `Compositions` - восстановленные workflows
- `Users` - пользователи системы
- `Datasets` - используемые датасеты

### Добавление нового эндпоинта

1. Создать роутер в `app/routers/`
2. Создать сервис в `app/services/`
3. Зарегистрировать роутер в `main.py`

Пример: см. [docs/development.md](docs/development.md)

## 📝 Changelog

### v2.1.0 (2025-10-19) - Sequential DAGNN рекомендации

- ✅ 🔮 **Sequential рекомендации** на основе Graph Neural Networks
- ✅ 🧠 **DAGNN алгоритм** для предсказания следующего сервиса/таблицы в workflow
- ✅ 🔗 **DAG-based** - использует структуру композиций из recover_new()
- ✅ 🎯 **Strict continuation** - только существующие связи в DAG
- ✅ 📊 **Table-based режим** - анализ только таблиц (игнорирует промежуточные сервисы)
- ✅ 📏 **Умная оценка** - учитывает distance, frequency и ML predictions
- ✅ 📦 **PyTorch + PyTorch Geometric** - современный deep learning
- ✅ 🔄 **Автообновление** - обучение в /update/full
- ✅ 💾 **Сохранение моделей** - быстрая загрузка при старте

### v2.0.0 (2025-10-12) - Новая система рекомендаций

- ✅ 🚀 **Новая архитектура рекомендаций** с множественными алгоритмами
- ✅ ⚡ **40-100x ускорение** рекомендаций (10-50ms вместо 2-5s)
- ✅ 💾 **LRU кэш** с TTL для мгновенных повторных запросов
- ✅ 🤖 **Умный выбор алгоритма** на основе профиля пользователя
- ✅ 🔧 **Модульная архитектура** - легко добавлять новые алгоритмы
- ✅ 📊 **Встроенный мониторинг** и статистика системы
- ✅ 🔄 **Обратная совместимость** со старыми эндпоинтами
- ✅ 🎯 **ids_only режим** - гибкий формат ответа

### v1.0.0 (2025-10-10)

- ✅ Полная миграция с Node.js на Python/FastAPI
- ✅ Поддержка композиций сервиса mapcombine (ID: 399)
- ✅ Улучшенная обработка WMS-сервисов
- ✅ Реорганизация статических файлов в `app/static/`
- ✅ Обновленная документация

## 🤝 Поддержка

По вопросам обращайтесь:
- 📧 Email: support@icc.ru
- 🌐 Web: http://geos.icc.ru:6868

## 📄 Лицензия

Proprietary - © 2025 ICC RU
