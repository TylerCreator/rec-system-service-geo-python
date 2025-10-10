# Service Recommendation System

> Система рекомендаций сервисов на основе анализа истории вызовов и композиций workflows

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)](https://www.postgresql.org/)

## 📋 Описание

API-сервис для системы рекомендаций геопространственных сервисов. Система:

- 🔍 **Анализирует** историю вызовов сервисов пользователями
- 🔗 **Восстанавливает** композиции сервисов (workflows) из истории
- 🤖 **Рекомендует** сервисы на основе машинного обучения (KNN)
- 📊 **Собирает** статистику использования
- ⚡ **Автоматически обновляет** данные по расписанию (cron)

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

## 🎯 Основные эндпоинты

После запуска доступны:

- **📚 API Docs (Swagger):** http://localhost:6868/docs
- **📖 ReDoc:** http://localhost:6868/redoc
- **❤️ Health Check:** http://localhost:6868/

### Ключевые API методы

```bash
# Статистика сервисов
GET /services/statistics

# Популярные сервисы
GET /services/popular

# Рекомендации для пользователя
GET /services/recomendation/{user_id}

# Восстановление композиций
GET /compositions/recover

# Полное обновление данных
GET /update/full
```

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
│   │   └── update.py
│   ├── services/                # Бизнес-логика
│   │   ├── calls_service.py
│   │   ├── compositions_service.py
│   │   └── update_service.py
│   └── static/                  # Статические файлы
│       ├── calls.csv
│       ├── compositionsDAG.json
│       ├── recomendations.json
│       ├── knn.py              # ML модель
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
# База данных
DB_HOST=localhost
DB_PORT=5431
DB_USER=postgres
DB_PASSWORD=postgres123
DB_NAME=rec_system

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
curl http://localhost:6868/update/statistics

# Восстановление композиций
curl http://localhost:6868/compositions/recover
```

## 🤖 Машинное обучение

Система использует алгоритм k-ближайших соседей (KNN) для рекомендаций:

1. Анализирует матрицу пользователь-сервис
2. Находит похожих пользователей
3. Рекомендует популярные сервисы среди похожих пользователей

Модель обучается автоматически при вызове:
```bash
python app/static/knn.py
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

# Получение статистики
curl http://localhost:6868/services/statistics

# Рекомендации для пользователя
curl http://localhost:6868/services/recomendation/50f7a1d80d58140037000006
```

## 📚 Документация

- **[Быстрый старт](docs/quickstart.md)** - детальная инструкция по запуску
- **[Конфигурация](docs/configuration.md)** - настройка приложения
- **[Развертывание](docs/deployment.md)** - деплой на сервер
- **[Docker](docs/docker.md)** - работа с контейнерами
- **[API](docs/api.md)** - описание эндпоинтов
- **[Миграция](docs/migration.md)** - переход с Node.js версии

## 🛠 Технологии

- **[FastAPI](https://fastapi.tiangolo.com/)** - современный веб-фреймворк
- **[SQLAlchemy](https://www.sqlalchemy.org/)** - ORM для работы с БД
- **[PostgreSQL](https://www.postgresql.org/)** - реляционная база данных
- **[Scikit-learn](https://scikit-learn.org/)** - машинное обучение
- **[APScheduler](https://apscheduler.readthedocs.io/)** - планировщик задач
- **[Docker](https://www.docker.com/)** - контейнеризация

## 🔧 Разработка

### Требования

- Python 3.9+
- PostgreSQL 13+
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
