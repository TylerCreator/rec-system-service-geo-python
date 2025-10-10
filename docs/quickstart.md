# 🚀 Быстрый старт

Подробная инструкция по запуску Service Recommendation System.

## Требования

- **Docker** и **Docker Compose** (рекомендуется)
- **ИЛИ** Python 3.9+, PostgreSQL 13+

## Вариант 1: Запуск с Docker (рекомендуется)

### Шаг 1: Подготовка

```bash
# Клонировать репозиторий (если еще не сделано)
cd rec-system-services-geo-python

# Скопировать файл с настройками
cp env.example .env
```

### Шаг 2: Настройка переменных окружения

Отредактируйте `.env` если нужны другие настройки:

```bash
nano .env
```

Основные параметры:
```bash
# Порты
NODE_LOCAL_PORT=6868        # Внешний порт для доступа
POSTGRESDB_LOCAL_PORT=5431  # Порт PostgreSQL

# База данных
POSTGRESDB_USER=postgres
POSTGRESDB_ROOT_PASSWORD=postgres123
POSTGRESDB_DATABASE=rec_system

# Приложение
DEBUG=false
ENABLE_CRON=true
```

### Шаг 3: Запуск

```bash
# Собрать и запустить все сервисы
docker-compose up -d --build

# Дождаться запуска (~30 секунд)
# Проверить статус
docker-compose ps
```

Вы должны увидеть два запущенных контейнера:
- `rec-system-services-geo-python-app-1` (приложение)
- `rec-system-services-geo-python-postgresdb-1` (база данных)

### Шаг 4: Проверка

Откройте в браузере:
- **API:** http://localhost:6868/
- **Swagger UI:** http://localhost:6868/docs
- **ReDoc:** http://localhost:6868/redoc

Или через curl:
```bash
# Проверка здоровья
curl http://localhost:6868/

# Должен вернуть:
# {"message":"Service Recommendation System API","status":"running"}
```

### Шаг 5: Инициализация данных

При первом запуске база данных пустая. Запустите обновление:

```bash
# Полное обновление (это займет несколько минут)
curl http://localhost:6868/update/full

# Или через Swagger UI:
# http://localhost:6868/docs -> /update/full -> Try it out -> Execute
```

Это загрузит данные из внешнего API (CRIS) и создаст:
- Список сервисов
- Историю вызовов (экспорт в CSV)
- Композиции сервисов
- Статистику
- Рекомендации

### Просмотр логов

```bash
# Логи приложения
docker-compose logs -f app

# Логи базы данных
docker-compose logs -f postgresdb
```

### Управление

```bash
# Остановить
docker-compose down

# Перезапустить
docker-compose restart

# Пересобрать и запустить
docker-compose up -d --build

# Удалить контейнеры и volumes
docker-compose down -v
```

## Вариант 2: Запуск без Docker

### Шаг 1: Установка PostgreSQL

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib

# macOS (с Homebrew)
brew install postgresql
brew services start postgresql

# Создать пользователя и базу данных
sudo -u postgres psql -c "CREATE USER postgres WITH PASSWORD 'postgres123';"
sudo -u postgres psql -c "CREATE DATABASE rec_system OWNER postgres;"
# Или если вы уже под пользователем postgres:
createdb rec_system
```

### Шаг 2: Установка Python зависимостей

```bash
# Создать виртуальное окружение
python -m venv venv

# Активировать
source venv/bin/activate  # Linux/macOS
# или
venv\Scripts\activate     # Windows

# Установить зависимости
pip install -r requirements.txt
```

### Шаг 3: Настройка окружения

```bash
# Скопировать env.example
cp env.example .env

# Отредактировать для локального запуска
nano .env
```

Настройки для локального запуска:
```bash
# База данных (локальная)
DB_HOST=localhost
DB_PORT=5432              # Стандартный порт PostgreSQL
DB_USER=postgres
DB_PASSWORD=postgres123
DB_NAME=rec_system

# Приложение
PORT=8080
DEBUG=true
ENABLE_CRON=true

# Файлы
CSV_FILE_PATH=app/static/calls.csv
RECOMMENDATIONS_FILE_PATH=app/static/recomendations.json
KNN_SCRIPT_PATH=app/static/knn.py
```

### Шаг 4: Запуск

```bash
# Запустить приложение
python main.py

# Приложение запустится на http://localhost:8080
```

### Шаг 5: Инициализация данных

```bash
# В другом терминале
curl http://localhost:8080/update/full
```

## Тестовые запросы

После инициализации данных попробуйте:

```bash
# Статистика сервисов
curl http://localhost:6868/services/statistics

# Топ популярных сервисов
curl http://localhost:6868/services/popular

# Рекомендации для пользователя
curl http://localhost:6868/services/recomendation/50f7a1d80d58140037000006

# Восстановление композиций
curl http://localhost:6868/compositions/recover

# Экспорт вызовов в CSV
curl http://localhost:6868/calls/dump-csv
```

## Автоматическое обновление

По умолчанию (с `ENABLE_CRON=true`) система автоматически:

- **Каждый час:** обновляет статистику сервисов
- **Каждые 6 часов:** пересчитывает рекомендации KNN

Вы можете отключить это, установив `ENABLE_CRON=false`.

## Решение проблем

### Порт уже занят

Если порт 6868 занят, измените `NODE_LOCAL_PORT` в `.env`:
```bash
NODE_LOCAL_PORT=8080
```

### База данных не запускается

```bash
# Проверить логи
docker-compose logs postgresdb

# Пересоздать volume
docker-compose down -v
docker-compose up -d
```

### Приложение не подключается к БД

Подождите ~30 секунд после запуска `docker-compose up` - база данных инициализируется.

Или проверьте настройки в `.env`:
```bash
DB_HOST=postgresdb  # для Docker
# или
DB_HOST=localhost   # для локального запуска
```

### CORS ошибки

Если нужно разрешить CORS для frontend, добавьте домен в `main.py`:
```python
origins = [
    "http://localhost:3000",
    "https://your-frontend.com"
]
```

## Следующие шаги

- 📚 Изучите [API документацию](api.md)
- ⚙️ Настройте [конфигурацию](configuration.md)
- 🚀 Разверните на [production сервере](deployment.md)
- 🔧 Узнайте про [разработку](development.md)

