# 🔌 Информация о портах

## Как работают порты в Docker

Docker использует **маппинг портов** для связи между хостом и контейнером:

```
HOST_PORT:CONTAINER_PORT
Внешний:Внутренний
```

---

## 📊 Настройка портов в проекте

### Приложение (FastAPI):

```bash
NODE_LOCAL_PORT=6868       # Внешний порт (доступ с хоста)
NODE_DOCKER_PORT=8080      # Внутренний порт (внутри контейнера)
```

**Это означает:**
- Приложение внутри Docker слушает порт `8080`
- Снаружи (с вашего компьютера) доступен на порту `6868`
- Доступ: `http://localhost:6868`

### База данных (PostgreSQL):

```bash
POSTGRESDB_LOCAL_PORT=5431      # Внешний порт (доступ с хоста)
POSTGRESDB_DOCKER_PORT=5432     # Внутренний порт (внутри контейнера)
```

**Это означает:**
- PostgreSQL внутри Docker слушает стандартный порт `5432`
- Снаружи доступен на порту `5431` (чтобы не конфликтовать с локальным PostgreSQL)
- Подключение с хоста: `localhost:5431`

---

## 🎯 Примеры использования

### Локальная разработка (docker-compose.yml):

**`.env` настройки:**
```bash
NODE_LOCAL_PORT=6868
NODE_DOCKER_PORT=8080
POSTGRESDB_LOCAL_PORT=5431
POSTGRESDB_DOCKER_PORT=5432
```

**Запуск:**
```bash
docker-compose up -d
```

**Доступ:**
- API: http://localhost:6868
- Swagger: http://localhost:6868/docs
- PostgreSQL: localhost:5431

### Production сервер (docker-compose-v2.yml):

**`.env` настройки (одинаковые с локальной разработкой):**
```bash
NODE_LOCAL_PORT=6868        # Внешний порт на сервере
NODE_DOCKER_PORT=8080       # Внутренний порт
POSTGRESDB_LOCAL_PORT=5432
POSTGRESDB_DOCKER_PORT=5432
```

**Запуск:**
```bash
docker-compose -f docker-compose-v2.yml up -d
```

**Доступ:**
- API: https://geos.icc.ru:6868 (порт 6868 с SSL)
- PostgreSQL: только внутри Docker сети

---

## 🔍 Почему порты отличаются?

### Преимущества разных портов:

1. **Избежание конфликтов:**
   - Если на вашем компьютере уже запущен PostgreSQL на порту 5432
   - Docker PostgreSQL использует 5431 снаружи, не будет конфликта

2. **Запуск нескольких проектов:**
   - Можно запустить несколько Docker проектов одновременно
   - Каждый на своем внешнем порту

3. **Безопасность:**
   - На production можно не открывать базу данных наружу
   - Оставить только HTTPS (443) для приложения

4. **Гибкость:**
   - Внутренние порты остаются стандартными (8080, 5432)
   - Внешние можно менять под ваши нужды

---

## 📝 Как изменить порты?

### Для локальной разработки:

Отредактируйте `.env`:
```bash
# Если порт 6868 занят, используйте другой
NODE_LOCAL_PORT=7070        # Ваш свободный порт
NODE_DOCKER_PORT=8080       # Не меняйте (внутренний)

# Если порт 5431 занят
POSTGRESDB_LOCAL_PORT=5433  # Ваш свободный порт
POSTGRESDB_DOCKER_PORT=5432 # Не меняйте (внутренний)
```

Перезапустите:
```bash
docker-compose down
docker-compose up -d
```

### Для production сервера:

Обычно используется стандартный HTTPS порт:
```bash
NODE_LOCAL_PORT=443         # Стандартный HTTPS
NODE_DOCKER_PORT=8080       # Внутренний
```

---

## 🛠️ Проверка портов

### Узнать какие порты заняты:

**Linux/Mac:**
```bash
# Проверить конкретный порт
lsof -i :6868

# Посмотреть все слушающие порты
netstat -tlnp | grep LISTEN
```

**Windows:**
```bash
netstat -ano | findstr :6868
```

### Проверить маппинг портов Docker:

```bash
# Посмотреть порты контейнеров
docker-compose ps

# Или
docker ps

# Детально о портах конкретного контейнера
docker port <container_name>
```

---

## 🔄 Схема работы

### Запрос к API:

```
Ваш браузер
    ↓
http://localhost:6868/docs
    ↓
Docker маппинг (6868 → 8080)
    ↓
FastAPI контейнер (порт 8080)
    ↓
Ответ
```

### Подключение к БД:

```
Python приложение (локально)
    ↓
postgresql://localhost:5431/rec_system
    ↓
Docker маппинг (5431 → 5432)
    ↓
PostgreSQL контейнер (порт 5432)
    ↓
База данных
```

### Внутри Docker (между контейнерами):

```
FastAPI контейнер
    ↓
postgresql://postgresdb:5432/rec_system
    ↓
PostgreSQL контейнер
    ↓
База данных

Примечание: Используется имя сервиса (postgresdb) и внутренний порт (5432)
```

---

## ⚠️ Важно помнить

1. **Внутренние порты (в контейнере):**
   - Обычно стандартные: 8080, 5432, 80, 443
   - Не меняйте без необходимости

2. **Внешние порты (на хосте):**
   - Можно менять свободно
   - Должны быть свободны на вашей системе

3. **В docker-compose.yml:**
   ```yaml
   ports:
     - "${NODE_LOCAL_PORT}:${NODE_DOCKER_PORT}"
   # Формат: ВНЕШНИЙ:ВНУТРЕННИЙ
   ```

4. **Переменные окружения внутри контейнера:**
   ```yaml
   environment:
     - PORT=${NODE_DOCKER_PORT}  # Используйте внутренний порт!
   ```

---

## 🎓 Примеры конфигураций

### Конфигурация 1: Разработка (порты отличаются)

**Преимущество:** Нет конфликтов с локальными сервисами

```bash
# .env
NODE_LOCAL_PORT=6868
NODE_DOCKER_PORT=8080
POSTGRESDB_LOCAL_PORT=5431
POSTGRESDB_DOCKER_PORT=5432
```

### Конфигурация 2: Production (одинаковые с разработкой)

**Преимущество:** Одинаковые порты для локальной разработки и production

```bash
# .env (одинаковый для локальной и production)
NODE_LOCAL_PORT=6868        # Одинаковый внешний порт
NODE_DOCKER_PORT=8080       # Одинаковый внутренний порт
POSTGRESDB_LOCAL_PORT=5431
POSTGRESDB_DOCKER_PORT=5432
```

### Конфигурация 3: Несколько проектов

**Преимущество:** Можно запустить параллельно

```bash
# Проект 1
NODE_LOCAL_PORT=6868
POSTGRESDB_LOCAL_PORT=5431

# Проект 2
NODE_LOCAL_PORT=6869
POSTGRESDB_LOCAL_PORT=5433

# Проект 3
NODE_LOCAL_PORT=6870
POSTGRESDB_LOCAL_PORT=5435
```

---

## 📞 Troubleshooting

### Ошибка: "Port already in use"

```bash
# 1. Найти что использует порт
lsof -i :6868  # Mac/Linux
netstat -ano | findstr :6868  # Windows

# 2. Либо остановить процесс, либо изменить порт в .env
NODE_LOCAL_PORT=7070

# 3. Перезапустить Docker
docker-compose down
docker-compose up -d
```

### Не могу подключиться к API

```bash
# Проверить что контейнер запущен
docker-compose ps

# Проверить логи
docker-compose logs app

# Проверить порты
docker ps

# Попробовать другой адрес
curl http://localhost:6868/
curl http://0.0.0.0:6868/
curl http://127.0.0.1:6868/
```

---

**Текущая конфигурация проекта (одинаковая для локальной и production):**

- 🌐 **API:** localhost:6868 → container:8080 (локально) / geos.icc.ru:6868 → container:8080 (production)
- 🗄️ **PostgreSQL:** localhost:5431 → container:5432
- 📖 **Swagger UI:** http://localhost:6868/docs (локально) / https://geos.icc.ru:6868/docs (production)

---

**Последнее обновление:** 9 октября 2025

