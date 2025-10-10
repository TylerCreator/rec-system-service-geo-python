# Быстрый старт

## 🚀 Запуск за 3 минуты

### 1. Подготовка

```bash
# Скопировать файл с примерами переменных окружения
cp env.example .env

# Опционально: отредактировать .env если нужны другие настройки
nano .env
```

### 2. Запуск с Docker (рекомендуется)

```bash
# Собрать и запустить все сервисы
docker-compose up -d --build

# Подождать несколько секунд пока база данных запустится
# Проверить статус
docker-compose ps

# Посмотреть логи (опционально)
docker-compose logs -f app
```

### 3. Проверка

Откройте в браузере:
- **API:** http://localhost:8080/
- **Документация (Swagger):** http://localhost:8080/docs
- **Документация (ReDoc):** http://localhost:8080/redoc

### 4. Тестовые запросы

```bash
# Здоровье приложения
curl http://localhost:8080/

# Получить список сервисов (будет пустой пока не обновите данные)
curl http://localhost:8080/services/

# Получить популярные сервисы
curl http://localhost:8080/services/popular

# Запустить полное обновление данных (это займет время!)
curl http://localhost:8080/update/full
```

## 📋 Основные команды

### Docker Compose

```bash
# Запустить
docker-compose up -d

# Остановить
docker-compose down

# Перезапустить
docker-compose restart

# Посмотреть логи
docker-compose logs -f app

# Пересобрать образ
docker-compose up -d --build

# Удалить все (включая volumes)
docker-compose down -v
```

### Локальный запуск (без Docker)

```bash
# 1. Установить PostgreSQL локально и создать БД
createdb rec_system

# 2. Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или venv\Scripts\activate на Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Настроить .env (изменить DB_HOST на localhost)
cp env.example .env
# Отредактировать DB_HOST=localhost в .env

# 5. Запустить
python main.py

# Или с автоперезагрузкой (для разработки)
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

## 🔧 Первоначальная настройка данных

После первого запуска база данных будет пустой. Нужно загрузить данные:

```bash
# 1. Обновить сервисы
curl http://localhost:8080/datasets/update
curl http://localhost:8080/calls/update-calls  # Это займет время!

# 2. Создать CSV файл
curl http://localhost:8080/calls/dump-csv

# 3. Обновить статистику
curl http://localhost:8080/update/statistic

# 4. Сгенерировать рекомендации
curl http://localhost:8080/update/recomendations

# 5. Восстановить композиции
curl http://localhost:8080/compositions/recover

# Или просто запустить полное обновление:
curl http://localhost:8080/update/full
```

⚠️ **Внимание:** Полное обновление может занять 10-30 минут в зависимости от объема данных!

## 📊 Основные эндпоинты

### Информация и документация
- `GET /` - Health check
- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc

### Сервисы
- `GET /services/` - Список всех сервисов
- `GET /services/popular?limit=20&type=any` - Популярные сервисы
- `GET /services/getRecomendation?user_id=USER` - Рекомендации для пользователя
- `GET /services/parameters/{service_id}` - История параметров сервиса

### Вызовы
- `GET /calls/` - История вызовов
- `GET /calls/update-calls` - Обновить вызовы
- `GET /calls/dump-csv` - Экспорт в CSV

### Композиции
- `GET /compositions/` - Все композиции
- `GET /compositions/recover` - Восстановить композиции
- `GET /compositions/stats` - Статистика

### Обновления
- `GET /update/full` - Полное обновление
- `GET /update/statistic` - Обновить статистику
- `GET /admin/run-cron` - Запустить cron вручную

## 🔍 Отладка

### Проблема: База данных не подключается

```bash
# Проверить что PostgreSQL запущен
docker-compose ps postgresdb

# Посмотреть логи PostgreSQL
docker-compose logs postgresdb

# Подключиться к БД напрямую
docker-compose exec postgresdb psql -U postgres -d rec_system
```

### Проблема: Приложение не запускается

```bash
# Посмотреть логи приложения
docker-compose logs app

# Проверить переменные окружения
docker-compose exec app env | grep DB_

# Перезапустить приложение
docker-compose restart app
```

### Проблема: Порт уже занят

```bash
# Проверить что занимает порт 8080
lsof -i :8080  # Mac/Linux
netstat -ano | findstr :8080  # Windows

# Изменить порт в .env
NODE_LOCAL_PORT=8081

# Перезапустить
docker-compose up -d
```

### Проблема: KNN скрипт не работает

```bash
# Проверить что файл существует
ls -la calls.csv

# Проверить что в нем есть данные
head calls.csv

# Запустить скрипт вручную
python3 knn.py calls.csv test_user

# Если нет CSV, создать его
curl http://localhost:8080/calls/dump-csv
```

## ⏰ Автоматические обновления

По умолчанию система автоматически обновляет данные каждый день в 00:00 (Иркутское время).

**Отключить автообновления:**
```bash
# В .env файле установить
ENABLE_CRON=false

# Перезапустить
docker-compose restart app
```

**Запустить обновление вручную:**
```bash
curl http://localhost:8080/admin/run-cron
```

## 🎯 Следующие шаги

1. ✅ Запустить приложение
2. ✅ Открыть документацию `/docs`
3. ✅ Загрузить начальные данные
4. 📖 Прочитать полную документацию в `README.new.md`
5. 📚 Изучить API endpoints в Swagger
6. 🔧 Настроить автоматические обновления
7. 🚀 Использовать API!

## 💡 Полезные ссылки

- **Swagger UI:** http://localhost:8080/docs
- **ReDoc:** http://localhost:8080/redoc
- **Полная документация:** README.new.md
- **Руководство по миграции:** MIGRATION_GUIDE.md

## ❓ Нужна помощь?

1. Проверьте логи: `docker-compose -f docker-compose.new.yml logs`
2. Прочитайте полную документацию в `README.new.md`
3. Посмотрите примеры в Swagger UI: http://localhost:8080/docs
4. Проверьте руководство по миграции: `MIGRATION_GUIDE.md`

---

**Готово!** 🎉 Приложение запущено и готово к использованию!

