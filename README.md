# Service Recommendation System

> Система рекомендаций сервисов на основе анализа истории вызовов и композиций workflows

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)](https://www.postgresql.org/)

##  Описание

API-сервис для системы рекомендаций геопространственных сервисов. Система:

-  **Анализирует** историю вызовов сервисов пользователями
-  **Восстанавливает** композиции сервисов (workflows) из истории
-  **Рекомендует** сервисы на основе ML (KNN, popularity-based + расширяемо)
-  **Быстрые рекомендации** с кэшированием (10-50ms против 2-5s)
-  **Собирает** статистику использования
-  **Автоматически обновляет** данные по расписанию (cron)

##  Быстрый старт

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

## 🎯 Основные эндпоинты

После запуска доступны:

- ** API Docs (Swagger):** http://localhost:6868/docs
- ** ReDoc:** http://localhost:6868/redoc
- ** Health Check:** http://localhost:6868/

### Ключевые API методы

```bash
#  Рекомендации для пользователя (новый API v2)
GET /services/recommendations/{user_id}?n=10&algorithm=knn

# Пакетные рекомендации
POST /services/recommendations/batch

# Статистика системы рекомендаций
GET /services/recommendations/stats

#  Последовательные рекомендации (workflow)
POST /sequential/predict         # Предсказать следующий сервис
POST /sequential/possible        # Возможные следующие сервисы (из DAG)
POST /sequential/tables/predict  # Предсказать следующую таблицу
POST /sequential/tables/possible # Возможные следующие таблицы (из DAG)
POST /sequential/train           # Обучить модель

# Популярные сервисы
GET /services/popular

# Восстановление композиций
GET /compositions/recover

# Полное обновление данных
GET /update/full

#  Legacy эндпоинты (deprecated)
GET /services/legacy/getRecomendations?user_id={id}
GET /services/legacy/getRecomendation?user_id={id}
```

>  **Совет:** Используйте новый API `/services/recommendations/{user_id}` вместо устаревшего `/services/legacy/getRecomendations`. Новый API в 40-100 раз быстрее!

##  Структура проекта

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
│   │   └── sequential.py       #  Sequential recommendations
│   ├── services/                # Бизнес-логика
│   │   ├── calls_service.py
│   │   ├── compositions_service.py
│   │   ├── recommendations_service.py  #  Новый сервис
│   │   ├── sequential_recommendations_service.py  #  Sequential сервис
│   │   ├── recommendations/     #  Модульная архитектура ML
│   │   │   ├── engine.py       # Движок рекомендаций
│   │   │   ├── data_loader.py  # Загрузка данных с кэшем
│   │   │   ├── cache.py        # LRU кэш
│   │   │   ├── algorithms/     # Алгоритмы рекомендаций
│   │   │   │   ├── knn.py                  # KNN collaborative filtering
│   │   │   │   ├── popularity.py           # Popularity-based
│   │   │   │   ├── analytics_popularity.py # Real-time analytics
│   │   │   │   └── sequential_dagnn.py     #  DAGNN для workflow
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

##  Конфигурация

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

> ** Важно для production:** Замените `root` на сильный пароль (минимум 32 символа) в обоих местах: `POSTGRESDB_ROOT_PASSWORD` и `DB_PASSWORD`!

Подробнее: [docs/configuration.md](docs/configuration.md)

##  Docker

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

##  Обновление данных

Система автоматически обновляет данные по расписанию:

- **Статистика:** каждый час
- **Рекомендации:** каждые 6 часов
- **Композиции:** при необходимости

Ручное обновление:

```bash
# Полное обновление всех данных
curl http://localhost:6868/update/full

# Только статистика
curl http://localhost:6868/update/statistics

# Восстановление композиций
curl http://localhost:6868/compositions/recover
```

##  Машинное обучение

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

#### 3. **Sequential Recomendations (True Graph Models)**
- Предсказывает следующий сервис или таблицу на основе истории вызовов и структуры рабочих процессов (workflows).
- Поддерживает 3 алгоритма на выбор (параметр `model`):
  - **DAGNN** - базовый алгоритм на основе Graph Neural Networks.
  - **SR-GNN (Session-based Graph Neural Network)** - анализирует подкоманды (сессии графов) с использованием механизма внимания (Attention) на основе нативных матричных операций.
  - **DAGTransformer (GraphGPS)** - SOTA алгоритм, объединяющий глубокое понимание графов через Message Passing (`GINConv`) и глобальное внимание трансформера (`GPSConv`).
- **Два режима:**
  - Services: предсказание следующего сервиса
  - Tables: предсказание следующей таблицы (игнорирует промежуточные сервисы)
- **Используется для:** генерации сложных многоканальных сценариев с параллельным ветвлением.

#### Ключевые преимущества:
-  **Быстро:** 10-50ms вместо 2-5 секунд
-  **Кэш:** LRU cache с TTL для мгновенных повторных запросов
- 🎯 **Умный выбор:** автоматический fallback для новых пользователей
- 🔧 **Расширяемо:** легко добавлять новые алгоритмы
-  **Метрики:** встроенный мониторинг и статистика

#### Примеры использования:

```bash
# Автоматический выбор алгоритма (рекомендуется)
curl "http://localhost:8080/services/recommendations/user123"

# Явный выбор алгоритма
curl "http://localhost:8080/services/recommendations/user123?algorithm=knn"

# Последовательные рекомендации (следующий сервис в workflow, по умолчанию dagnn)
curl -X POST "http://localhost:8080/sequential/predict" \
  -H "Content-Type: application/json" \
  -d '{"sequence": [123, 456, 789], "n": 5, "model": "dag-transformer"}'

# Последовательные рекомендации таблиц (следующая таблица в workflow)
curl -X POST "http://localhost:8080/sequential/tables/predict" \
  -H "Content-Type: application/json" \
  -d '{"table_sequence": [1002120, 1001211], "n": 5, "ids_only": true, "model": "sr-gnn"}'
```

#### Инициализация:
Модели автоматически инициализируются при старте приложения:
```bash
python main.py
#  Recommendation engine initialized
#  Sequential DAGNN engine initialized
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

##  Композиции сервисов

Система восстанавливает workflows из истории вызовов:

- Анализирует связи между вызовами через входные/выходные данные
- Строит граф композиций (DAG)
- Сохраняет в `app/static/compositionsDAG.json`

Особенности:
- Поддержка WMS-сервисов
- Поддержка сервиса mapcombine (ID: 399)
- Отслеживание промежуточных результатов через `edit` виджеты

##  Тестирование

```bash
# Проверка здоровья
curl http://localhost:6868/

# Получение статистики
curl http://localhost:6868/services/statistics

# Рекомендации для пользователя
curl http://localhost:6868/services/recomendation/50f7a1d80d58140037000006
```

##  Документация

### Основная
- **[Быстрый старт](docs/quickstart.md)** - детальная инструкция по запуску
- **[Конфигурация](docs/configuration.md)** - настройка приложения
- **[Развертывание](docs/deployment.md)** - деплой на сервер
- **[API](docs/api.md)** - описание эндпоинтов
- **[Миграция](docs/migration.md)** - переход с Node.js версии

###  Система рекомендаций
- **[Архитектура](docs/recommendations_architecture.md)** - дизайн новой системы
- **[Использование](docs/recommendations_usage.md)** - примеры и best practices
- **[Миграция на v2](docs/recommendations_migration.md)** - переход на новый API

##  Технологии

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

##  Changelog

### v2.2.0 (Текущая версия) - True Graph Models (SOTA GraphGPS & SR-GNN)
-   **GraphGPS (DAGTransformer)** - Интеграция State-of-the-Art архитектуры графовых трансформеров из PyTorch Geometric (`GPSConv`, `GINConv`). 
-   **SR-GNN (Session-based Graph Neural Network)** - Внедрена модель рекомендаций, основанная на сессионных графах с механизмом локального и глобального внимания (`Local/Global Intent`).
-   **Инкрементальные подграфы** - Переход от простых плоских последовательностей (Sequences) к честным топологическим графам (`target nodes`, `edge_index`, `Data`/`Batch` objects). Учитываются развилки и параллельные сценарии (`BCEWithLogitsLoss`).
-   **Мультимодельный роутинг** - Все эндпоинты (`/sequential/predict`, `/sequential/tables/predict`, `/sequential/train`) теперь принимают параметр `model` (`dagnn`, `sr-gnn`, `dag-transformer`). Обучение по Cron (в `/update/full`) последовательно тренирует все модели.
-   **Bugfixes** - Исправлен краш `matmul primitive` на CPU (ONEDNN) и спам из-за парсинга пустых JSON-строк.

### v2.1.0 (2025-10-19) - Sequential DAGNN рекомендации

-   **Sequential рекомендации** на основе Graph Neural Networks
-   **DAGNN алгоритм** для предсказания следующего сервиса/таблицы в workflow
-   **DAG-based** - использует структуру композиций из recover_new()
-  🎯 **Strict continuation** - только существующие связи в DAG
-   **Table-based режим** - анализ только таблиц (игнорирует промежуточные сервисы)
-   **Умная оценка** - учитывает distance, frequency и ML predictions
-   **PyTorch + PyTorch Geometric** - современный deep learning
-   **Автообновление** - обучение в /update/full
-   **Сохранение моделей** - быстрая загрузка при старте

### v2.0.0 (2025-10-12) - Новая система рекомендаций

-   **Новая архитектура рекомендаций** с множественными алгоритмами
-   **40-100x ускорение** рекомендаций (10-50ms вместо 2-5s)
-   **LRU кэш** с TTL для мгновенных повторных запросов
-   **Умный выбор алгоритма** на основе профиля пользователя
-  🔧 **Модульная архитектура** - легко добавлять новые алгоритмы
-   **Встроенный мониторинг** и статистика системы
-   **Обратная совместимость** со старыми эндпоинтами
-  🎯 **ids_only режим** - гибкий формат ответа

### v1.0.0 (2025-10-10)

-  Полная миграция с Node.js на Python/FastAPI
-  Поддержка композиций сервиса mapcombine (ID: 399)
-  Улучшенная обработка WMS-сервисов
-  Реорганизация статических файлов в `app/static/`
-  Обновленная документация

##  Поддержка

По вопросам обращайтесь:
-  Email: support@icc.ru
-  Web: http://geos.icc.ru:6868

##  Лицензия

Proprietary - © 2025 ICC RU
