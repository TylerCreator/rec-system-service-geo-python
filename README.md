# Service Recommendation System (FastAPI)

Система рекомендаций сервисов на основе истории вызовов. Переписана с Node.js/Express на Python/FastAPI.

## Описание

Это API для системы рекомендаций сервисов, которая:
- Отслеживает вызовы сервисов пользователями
- Анализирует композиции сервисов (workflows)
- Предоставляет рекомендации на основе машинного обучения (KNN)
- Собирает статистику использования сервисов
- Автоматически обновляет данные по расписанию

## Технологический стек

- **FastAPI** - современный веб-фреймворк для Python
- **SQLAlchemy** - ORM для работы с базой данных
- **PostgreSQL** - реляционная база данных
- **asyncpg** - асинхронный драйвер PostgreSQL
- **Scikit-learn** - машинное обучение (KNN для рекомендаций)
- **APScheduler** - планировщик задач (cron jobs)
- **Docker & Docker Compose** - контейнеризация

## Структура проекта

```
.
├── main.py                      # Главный файл приложения
├── app/
│   ├── core/
│   │   ├── config.py           # Конфигурация приложения
│   │   └── database.py         # Подключение к БД
│   ├── models/
│   │   └── models.py           # SQLAlchemy модели
│   ├── routers/
│   │   ├── calls.py            # API для вызовов сервисов
│   │   ├── services.py         # API для сервисов
│   │   ├── datasets.py         # API для датасетов
│   │   ├── compositions.py     # API для композиций
│   │   └── update.py           # API для обновлений
│   └── services/
│       ├── calls_service.py
│       ├── services_service.py
│       ├── datasets_service.py
│       ├── compositions_service.py
│       └── update_service.py
├── knn.py                       # ML скрипт для рекомендаций
├── requirements.txt             # Python зависимости
├── Dockerfile.new              # Docker образ
├── docker-compose.new.yml      # Docker Compose конфигурация
└── .env.example                # Пример файла окружения
```

## Установка и запуск

### Локальный запуск

1. **Клонировать репозиторий и перейти в директорию:**

```bash
cd rec-system-services-geo-python
```

2. **Создать виртуальное окружение:**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

3. **Установить зависимости:**

```bash
pip install -r requirements.txt
```

4. **Настроить переменные окружения:**

```bash
cp .env.example .env
# Отредактировать .env файл с нужными настройками
```

5. **Запустить PostgreSQL** (если не используете Docker):

```bash
# Убедитесь, что PostgreSQL запущен и создана база данных
createdb rec_system
```

6. **Запустить приложение:**

```bash
python main.py
# или
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

7. **Открыть документацию API:**

- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

### Запуск с Docker Compose

1. **Создать .env файл:**

```bash
cp .env.example .env
```

2. **Собрать и запустить контейнеры:**

```bash
docker-compose -f docker-compose.new.yml up -d --build
```

3. **Проверить статус:**

```bash
docker-compose -f docker-compose.new.yml ps
docker-compose -f docker-compose.new.yml logs -f app
```

4. **Остановить контейнеры:**

```bash
docker-compose -f docker-compose.new.yml down
```

## API Endpoints

### Основные эндпоинты

- `GET /` - Health check
- `GET /docs` - Swagger UI документация
- `GET /redoc` - ReDoc документация

### Calls (Вызовы сервисов)

- `GET /calls/` - Получить все вызовы
- `GET /calls/update-calls` - Обновить вызовы с удаленного сервера
- `GET /calls/dump-csv` - Экспортировать вызовы в CSV

### Services (Сервисы)

- `GET /services/` - Получить все сервисы
- `GET /services/getRecomendations?user_id={id}` - Получить рекомендации (real-time)
- `GET /services/getRecomendation?user_id={id}` - Получить рекомендации (cached)
- `GET /services/popular` - Получить популярные сервисы
- `GET /services/parameters/{service_id}` - Получить параметры сервиса

### Datasets (Датасеты)

- `GET /datasets/update` - Обновить датасеты

### Compositions (Композиции)

- `GET /compositions/` - Получить все композиции
- `GET /compositions/recover` - Восстановить композиции из истории
- `GET /compositions/recoverNew` - Улучшенный алгоритм восстановления
- `GET /compositions/stats` - Статистика композиций

### Update (Обновления)

- `GET /update/all` - Обновить все данные
- `GET /update/full` - Полное обновление системы
- `GET /update/recomendations` - Обновить рекомендации
- `GET /update/statistic` - Обновить статистику
- `GET /update/local` - Локальное обновление (только статистика)

### Admin

- `GET /admin/run-cron` - Ручной запуск cron задачи

## Автоматические обновления

Приложение автоматически запускает полное обновление системы каждый день в 00:00 по времени Иркутска (Asia/Irkutsk).

Для отключения автоматических обновлений установите в `.env`:

```
ENABLE_CRON=false
```

## Конфигурация

Все настройки конфигурируются через переменные окружения в файле `.env`:

- **Основные настройки:**
  - `PORT` - порт приложения (по умолчанию 8080)
  - `DEBUG` - режим отладки (true/false)
  - `ENABLE_CRON` - включить автоматические обновления (true/false)

- **База данных:**
  - `DB_HOST` - хост PostgreSQL
  - `DB_PORT` - порт PostgreSQL
  - `DB_USER` - пользователь
  - `DB_PASSWORD` - пароль
  - `DB_NAME` - имя базы данных

- **SSL (опционально):**
  - `SSL_ENABLED` - включить HTTPS (true/false)
  - `SSL_KEY_PATH` - путь к приватному ключу
  - `SSL_CERT_PATH` - путь к сертификату

## Миграция с Node.js версии

Если вы мигрируете с предыдущей версии на Node.js:

1. **Данные в базе совместимы** - структура таблиц идентична
2. **Обновите Docker Compose:**

```bash
# Остановите старую версию
docker-compose down

# Запустите новую версию
docker-compose -f docker-compose.new.yml up -d
```

3. **Файлы данных** (calls.csv, recomendations.json) остаются совместимыми

## Разработка

### Форматирование кода

```bash
# Установить инструменты разработки
pip install black isort flake8

# Форматировать код
black .
isort .

# Проверить линтером
flake8 .
```

### Миграции базы данных

Для создания миграций используется Alembic:

```bash
# Инициализировать Alembic (первый раз)
alembic init alembic

# Создать миграцию
alembic revision --autogenerate -m "Description"

# Применить миграции
alembic upgrade head
```

## Производительность

- **Асинхронность** - все операции с БД и внешними API асинхронные
- **Connection pooling** - пул соединений с PostgreSQL (10-20 подключений)
- **Кеширование** - рекомендации сохраняются в файл для быстрого доступа
- **Batch операции** - обновления данных выполняются пакетами

## Мониторинг и логи

Приложение выводит структурированные логи:

```bash
# Просмотр логов в Docker
docker-compose -f docker-compose.new.yml logs -f app

# Фильтрация логов
docker-compose -f docker-compose.new.yml logs app | grep ERROR
```

## Troubleshooting

### База данных не подключается

```bash
# Проверить что PostgreSQL запущен
docker-compose -f docker-compose.new.yml ps postgresdb

# Проверить логи
docker-compose -f docker-compose.new.yml logs postgresdb

# Проверить подключение вручную
docker-compose -f docker-compose.new.yml exec postgresdb psql -U postgres -d rec_system
```

### Приложение не запускается

```bash
# Проверить логи
docker-compose -f docker-compose.new.yml logs app

# Перезапустить
docker-compose -f docker-compose.new.yml restart app

# Пересобрать образ
docker-compose -f docker-compose.new.yml up -d --build app
```

### KNN скрипт не работает

```bash
# Проверить что CSV файл существует
ls -la calls.csv

# Запустить скрипт вручную
python3 knn.py calls.csv <user_id>
```

## Лицензия

ISC

## Поддержка

Для вопросов и поддержки создайте issue в репозитории проекта.

## Changelog

### Version 2.0.0 (FastAPI Migration)

- ✅ Полная переработка с Node.js/Express на Python/FastAPI
- ✅ Асинхронные операции с базой данных
- ✅ Улучшенная производительность
- ✅ Автоматическая документация API (Swagger/ReDoc)
- ✅ Типизация с помощью Pydantic
- ✅ Сохранена совместимость с предыдущей версией
- ✅ Docker контейнеризация
- ✅ Планировщик задач (APScheduler)
- ✅ Машинное обучение (Scikit-learn KNN)

