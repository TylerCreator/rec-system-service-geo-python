# 🎯 Принципы конфигурации проекта

## Single Source of Truth - Все настройки в `.env`

---

## ✅ Правильный подход

### Все параметры читаются из `.env`:

```
.env файл (единый источник правды)
    ↓
Environment variables в Docker контейнере
    ↓
app/core/config.py (Settings класс с Pydantic)
    ↓
main.py читает settings
    ↓
Приложение запускается с правильными параметрами
```

### Что это означает:

**✅ В docker-compose.yml:**
```yaml
command: python main.py  # Просто запускаем приложение
```

**✅ В Dockerfile:**
```dockerfile
CMD ["python", "main.py"]  # Просто запускаем приложение
```

**✅ В .env:**
```bash
# Все настройки здесь
PORT=8080
SSL_ENABLED=true
SSL_KEY_PATH=/certs/privkey.pem
SSL_CERT_PATH=/certs/fullchain.pem
NODE_LOCAL_PORT=6868
NODE_DOCKER_PORT=8080
```

**✅ В main.py:**
```python
# Читаем из settings (которые из .env)
ssl_keyfile = settings.SSL_KEY_PATH if settings.SSL_ENABLED else None
ssl_certfile = settings.SSL_CERT_PATH if settings.SSL_ENABLED else None

uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=settings.PORT,
    ssl_keyfile=ssl_keyfile,
    ssl_certfile=ssl_certfile
)
```

---

## ❌ Неправильный подход (избегать)

### Дублирование параметров:

**❌ НЕ делайте так:**
```yaml
# В docker-compose.yml
command: python -m uvicorn main:app --host 0.0.0.0 --port 8080 --ssl-keyfile /certs/privkey.pem
```

**Почему плохо:**
1. 🔴 Дублирование настроек (и в .env, и в команде)
2. 🔴 Трудно поддерживать (нужно менять в нескольких местах)
3. 🔴 Легко ошибиться (разные значения в разных файлах)
4. 🔴 Не гибко (нельзя легко переключить HTTP/HTTPS)

---

## 💡 Преимущества текущего подхода

### 1. Централизованная конфигурация

Все настройки в одном месте - `.env` файле:
```bash
# Меняете здесь
SSL_ENABLED=true
PORT=8080

# Не нужно менять в docker-compose.yml, Dockerfile или где-то еще
```

### 2. Гибкость

```bash
# Переключить на HTTP - просто измените .env
SSL_ENABLED=false

# Изменить порт
PORT=9000

# Перезапустите - и всё работает!
docker-compose restart app
```

### 3. Одинаковая логика везде

```bash
# Локально
python main.py

# В Docker (docker-compose.yml)
python main.py

# В production (docker-compose-v2.yml)
python main.py

# Везде одинаково!
```

### 4. Умное поведение

`main.py` автоматически определяет:
- Есть ли SSL сертификаты → запускает HTTPS
- Нет сертификатов → запускает HTTP
- Включен ли DEBUG режим → добавляет auto-reload
- Все из переменных окружения!

---

## 🔄 Как это работает

### При запуске контейнера:

```
1. Docker Compose читает .env файл
   ↓
2. Передает переменные в контейнер как environment
   ↓
3. Python приложение запускается: python main.py
   ↓
4. main.py импортирует settings из app/core/config.py
   ↓
5. Settings (Pydantic) автоматически читает environment variables
   ↓
6. main.py использует settings для конфигурации uvicorn
   ↓
7. Приложение запускается с правильными параметрами
```

### Pример настройки SSL:

```python
# В app/core/config.py
class Settings(BaseSettings):
    SSL_ENABLED: bool = False
    SSL_KEY_PATH: Optional[str] = "/certs/privkey.pem"
    SSL_CERT_PATH: Optional[str] = "/certs/fullchain.pem"
    PORT: int = 8080

# В main.py
if settings.SSL_ENABLED and os.path.exists(settings.SSL_KEY_PATH):
    # Запуск с SSL
    uvicorn.run(..., ssl_keyfile=settings.SSL_KEY_PATH, ...)
else:
    # Запуск без SSL
    uvicorn.run(...)
```

---

## 📝 Чек-лист правильной конфигурации

- [x] ✅ Все настройки в `.env` файле
- [x] ✅ `main.py` читает настройки через `settings`
- [x] ✅ `Dockerfile` использует простой CMD: `python main.py`
- [x] ✅ `docker-compose.yml` использует простой command (или не указывает)
- [x] ✅ Нет жестко заданных параметров в командной строке
- [x] ✅ Нет дублирования конфигурации

---

## 🎯 Ответ на вопрос

**Да, текущий Dockerfile будет работать:**
- ✅ С docker-compose version 2
- ✅ С docker-compose version 3.8
- ✅ С docker-compose version 3.x
- ✅ Вообще с любой версией Docker Compose
- ✅ И даже без docker-compose (с обычным `docker run`)

**Версия docker-compose влияет только на:**
- Синтаксис docker-compose.yml файла
- Доступные функции (healthcheck, depends_on с condition, и т.д.)
- НЕ влияет на Dockerfile!

---

## 🚀 Как использовать

### С docker-compose version 2 (production):
```bash
docker-compose -f docker-compose-v2.yml up -d --build
# Dockerfile собирается один раз
# Все настройки из .env
```

### С docker-compose version 3.8 (локально):
```bash
docker-compose up -d --build
# Тот же Dockerfile
# Все настройки из .env
```

### Без docker-compose (чистый Docker):
```bash
docker build -t rec-system .
docker run -p 6868:8080 --env-file .env rec-system
# Тот же Dockerfile
# Все настройки из .env
```

---

## 💡 Итог

**Текущая конфигурация идеальна:**
- ✅ Dockerfile универсальный (работает везде)
- ✅ Все настройки в `.env` (Single Source of Truth)
- ✅ Нет дублирования
- ✅ Легко поддерживать
- ✅ Работает с любой версией Docker Compose

**Не нужно ничего менять!** 🎉

---

**Последнее обновление:** 9 октября 2025

