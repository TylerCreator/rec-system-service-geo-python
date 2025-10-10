# 🎯 НАЧНИТЕ ОТСЮДА

## Проект успешно мигрирован на Python/FastAPI! ✅

---

## 🚀 Что нужно сделать СЕЙЧАС:

### 1. Настройте окружение (30 секунд)

```bash
# Скопируйте файл с настройками
cp env.example .env

# Опционально: откройте и измените если нужно
nano .env
```

### 2. Запустите приложение (2 минуты)

**С Docker локально (HTTP):**
```bash
docker-compose up -d --build
```

**С Docker на production сервере (HTTPS):**
```bash
# На сервере с SSL сертификатами
docker-compose -f docker-compose-v2.yml up -d --build
```

**Или локально без Docker:**
```bash
python -m venv venv
source venv/bin/activate  # или venv\Scripts\activate на Windows
pip install -r requirements.txt
python main.py
```

### 3. Проверьте что работает (1 минута)

Откройте в браузере:
- **API:** http://localhost:6868/ (или http://localhost:8080 если не меняли порты)
- **Документация:** http://localhost:6868/docs

Должны увидеть:
```json
{
  "message": "Service Recommendation System API",
  "status": "running",
  "version": "2.0.0"
}
```

💡 **Примечание о портах:**
- По умолчанию в `env.example`: внешний порт **6868** → внутренний **8080**
- PostgreSQL: внешний порт **5431** → внутренний **5432**
- Подробнее: см. `PORTS_INFO.md`

### 4. Загрузите данные (опционально, ~10-30 минут)

```bash
# Полное обновление всех данных
curl http://localhost:8080/update/full

# Или по частям:
curl http://localhost:8080/calls/update-calls
curl http://localhost:8080/datasets/update
curl http://localhost:8080/calls/dump-csv
curl http://localhost:8080/update/statistic
```

---

## 📚 Что читать дальше:

1. **`QUICKSTART.md`** - быстрый старт и основные команды
2. **`README.md`** - полная документация проекта
3. **`DOCKER_COMPOSE_INFO.md`** - информация о Docker конфигурациях
4. **`MIGRATION_GUIDE.md`** - детали миграции с Node.js
5. **`MIGRATION_SUMMARY.md`** - что было сделано

---

## 📦 Варианты запуска:

### 1. Локально с HTTP (разработка):
```bash
docker-compose up -d --build
# Доступ: http://localhost:6868 (или :8080 если изменили порт в .env)
```

### 2. На сервере с HTTPS (production):
```bash
docker-compose -f docker-compose-v2.yml up -d --build
# Доступ: https://geos.icc.ru:6868 (или ваш домен)
```

### 3. Только БД (локальная разработка Python):
```bash
docker-compose -f docker-compose-db.yml up -d
python main.py
# Доступ: http://localhost:8080 (приложение запущено локально, не в Docker)
```

**📚 Подробнее:** См. `DOCKER_COMPOSE_INFO.md` для полной информации о конфигурациях

---

## 🔍 Проверочный список:

- [ ] Скопировал `env.example` в `.env`
- [ ] Запустил `docker-compose up -d --build` (локально) или `docker-compose -f docker-compose-v2.yml up -d --build` (на сервере)
- [ ] Открыл http://localhost:6868/docs (или https://geos.icc.ru:6868/docs на сервере)
- [ ] Увидел Swagger UI с документацией API
- [ ] (Опционально) Загрузил данные через `/update/full`

---

## 💡 Полезные команды:

```bash
# Посмотреть логи
docker-compose -f docker-compose.new.yml logs -f app

# Перезапустить
docker-compose -f docker-compose.new.yml restart app

# Остановить
docker-compose -f docker-compose.new.yml down

# Пересобрать
docker-compose -f docker-compose.new.yml up -d --build
```

---

## ❓ Проблемы?

### База данных не подключается
```bash
docker-compose logs postgresdb
docker-compose restart postgresdb
```

### Приложение не запускается
```bash
docker-compose logs app
docker-compose restart app
```

### Порт занят
```bash
# Измените порт в .env
NODE_LOCAL_PORT=8081
docker-compose up -d
```

---

## 📖 Документация API:

После запуска:
- **Swagger UI:** http://localhost:6868/docs
- **ReDoc:** http://localhost:6868/redoc
- **OpenAPI JSON:** http://localhost:6868/openapi.json

💡 **Порты:** Локально используется 6868 (внешний) → 8080 (внутри Docker)

В Swagger UI можно:
- Просмотреть все эндпоинты
- Протестировать запросы
- Посмотреть примеры ответов
- Увидеть схемы данных

---

## 🎯 Основные эндпоинты:

```bash
# Здоровье приложения
curl http://localhost:6868/

# Список сервисов
curl http://localhost:6868/services/

# Популярные сервисы
curl http://localhost:6868/services/popular?limit=10

# Рекомендации для пользователя
curl http://localhost:6868/services/getRecomendation?user_id=USER_ID

# Все композиции
curl http://localhost:6868/compositions/

# Полное обновление
curl http://localhost:6868/update/full
```

---

## 🔄 Миграция с Node.js версии:

Если у вас есть работающая Node.js версия:

1. **База данных совместима** - можно использовать ту же БД
2. **API совместим** - все эндпоинты те же
3. **Файлы совместимы** - calls.csv, recomendations.json

Просто замените:
```bash
# Остановите старую версию (если была Node.js версия)
docker-compose down

# Запустите новую Python версию
docker-compose up -d --build
```

---

## ✅ Всё работает? Отлично!

Теперь вы можете:
- 📊 Использовать API для получения данных
- 🤖 Получать ML рекомендации
- 📈 Анализировать статистику
- 🔄 Автоматически обновлять данные (cron)

---

## 📞 Нужна помощь?

1. Прочитайте `QUICKSTART.md`
2. Посмотрите логи: `docker-compose logs` (или `docker-compose -f docker-compose-v2.yml logs` на сервере)
3. Проверьте документацию: http://localhost:6868/docs
4. Прочитайте `README.md`, `DOCKER_COMPOSE_INFO.md` и `PORTS_INFO.md`

---

## 🎓 Что нового в Python версии:

- ✅ Асинхронные операции (быстрее)
- ✅ Автоматическая документация (Swagger)
- ✅ Типизация (type hints)
- ✅ Валидация данных (Pydantic)
- ✅ Меньше потребление памяти
- ✅ Лучшая поддержка IDE

---

## 📦 Структура файлов:

```
Python/FastAPI FILES:
├── main.py                      ← Главный файл
├── app/                         ← Код приложения
│   ├── core/                    ← Конфигурация и БД
│   ├── models/                  ← Модели SQLAlchemy
│   ├── routers/                 ← API эндпоинты
│   └── services/                ← Бизнес-логика
├── requirements.txt             ← Python зависимости
├── Dockerfile                   ← Docker образ
├── docker-compose.yml          ← Docker Compose (основной)
├── docker-compose-v2.yml       ← Docker Compose (альтернативный)
├── docker-compose-db.yml       ← Docker Compose (только БД)
├── env.example                  ← Пример настроек
├── knn.py                       ← ML скрипт рекомендаций
└── wait-for-it.sh              ← Скрипт ожидания БД

DOCUMENTATION:
├── START_HERE.md               ← Этот файл (начните отсюда)
├── QUICKSTART.md               ← Быстрый старт
├── README.md                   ← Полная документация
├── MIGRATION_GUIDE.md          ← Руководство по миграции
└── MIGRATION_SUMMARY.md        ← Итоги миграции
```

---

## 🚀 Готово к использованию!

**Следующий шаг:** Откройте http://localhost:6868/docs

💡 **О портах:** Проект использует порт **6868** снаружи и **8080** внутри Docker.  
Подробнее: `PORTS_INFO.md`

Удачи! 🎉

---

**Версия:** 2.0.0  
**Дата:** 9 октября 2025  
**Статус:** ✅ Production Ready

