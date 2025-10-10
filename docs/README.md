# 📚 Документация

Полная документация Service Recommendation System.

## 📖 Содержание

### Быстрый старт
- **[Quickstart Guide](quickstart.md)** - пошаговая инструкция по запуску системы

### Конфигурация и настройка
- **[Configuration](configuration.md)** - детальное описание всех настроек
- **[Environment Variables](configuration.md#переменные-окружения)** - переменные окружения

### Развертывание
- **[Deployment Guide](deployment.md)** - развертывание на production сервере
- **[Docker Setup](deployment.md#развертывание-с-docker-compose)** - работа с Docker

### API
- **[API Documentation](api.md)** - описание всех эндпоинтов
- **[Interactive Docs](http://localhost:6868/docs)** - Swagger UI (после запуска)

### Миграция
- **[Migration Guide](migration.md)** - переход с Node.js версии

## 🚀 Краткое руководство

### Запуск за 3 минуты

```bash
# 1. Настройка
cp env.example .env

# 2. Запуск
docker-compose up -d --build

# 3. Проверка
curl http://localhost:6868/docs
```

### Основные команды

```bash
# Остановить
docker-compose down

# Просмотр логов
docker-compose logs -f app

# Обновить данные
curl http://localhost:6868/update/full

# Восстановить композиции
curl http://localhost:6868/compositions/recover
```

## 📁 Структура документации

```
docs/
├── README.md           # Этот файл
├── quickstart.md       # Быстрый старт
├── configuration.md    # Конфигурация
├── deployment.md       # Развертывание
├── api.md             # API документация
├── migration.md        # Миграция с Node.js
└── archive/           # Архив старых документов
    ├── MIGRATION_GUIDE.md
    ├── SERVER_DEPLOYMENT.md
    └── ...
```

## 🎯 По задачам

### Первый запуск
1. Прочитайте [Quickstart](quickstart.md)
2. Настройте [конфигурацию](configuration.md)
3. Запустите систему

### Разработка
1. Изучите [API](api.md)
2. Ознакомьтесь со структурой проекта в [README.md](../README.md)
3. Настройте окружение для разработки

### Production
1. Следуйте [Deployment Guide](deployment.md)
2. Настройте SSL сертификаты
3. Настройте мониторинг и backup

### Миграция с Node.js
1. Прочитайте [Migration Guide](migration.md)
2. Сделайте backup данных
3. Следуйте пошаговым инструкциям

## 🔗 Полезные ссылки

### Внешние ресурсы
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy ORM](https://docs.sqlalchemy.org/)
- [PostgreSQL Docs](https://www.postgresql.org/docs/)
- [Docker Documentation](https://docs.docker.com/)

### Интерактивная документация
- **Swagger UI:** http://localhost:6868/docs
- **ReDoc:** http://localhost:6868/redoc

## ❓ Частые вопросы

### Как запустить без Docker?

См. [Quickstart - Вариант 2](quickstart.md#вариант-2-запуск-без-docker)

### Как изменить порт?

Измените `NODE_LOCAL_PORT` в `.env`. См. [Configuration](configuration.md#docker-порты)

### Как получить SSL сертификаты?

См. [Deployment - HTTPS Setup](deployment.md#вариант-2-https-с-ssl)

### Как обновить данные?

```bash
curl http://localhost:6868/update/full
```

См. [API - Update Endpoints](api.md#update)

### Как мигрировать с Node.js?

Следуйте [Migration Guide](migration.md)

## 📝 Changelog

### v1.0.0 (2025-10-10)
- ✅ Полная миграция на Python/FastAPI
- ✅ Поддержка композиций mapcombine (ID: 399)
- ✅ Улучшенная обработка WMS-сервисов
- ✅ Реорганизация документации

## 🆘 Поддержка

### Проблемы и вопросы

1. Проверьте [Quickstart](quickstart.md#решение-проблем)
2. Проверьте [Deployment](deployment.md#решение-проблем)
3. Изучите логи: `docker-compose logs -f app`

### Контакты

- 📧 Email: support@icc.ru
- 🌐 Web: http://geos.icc.ru:6868

## 📄 Архив

Старые версии документации находятся в `docs/archive/`:
- `MIGRATION_GUIDE.md` - оригинальное руководство по миграции
- `SERVER_DEPLOYMENT.md` - старая инструкция по развертыванию
- `CONFIGURATION_PRINCIPLES.md` - принципы конфигурации
- И другие...

Эти файлы сохранены для справки, но могут содержать устаревшую информацию.

