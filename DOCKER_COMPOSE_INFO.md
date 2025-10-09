# Docker Compose конфигурации

## Обзор

В проекте есть три Docker Compose файла для разных сценариев использования:

---

## 1. `docker-compose.yml` (основной)

**Назначение:** Локальная разработка и тестирование

**Особенности:**
- ✅ Version 3.8 (современный синтаксис)
- ✅ Healthchecks для БД
- ✅ wait-for-it.sh для синхронизации запуска
- ✅ HTTP режим (без SSL)
- ✅ Порт по умолчанию: 8080
- ✅ Опциональное монтирование кода для разработки

**Использование:**
```bash
# Запуск
docker-compose up -d --build

# Логи
docker-compose logs -f app

# Остановка
docker-compose down
```

**Когда использовать:**
- Локальная разработка
- Тестирование
- CI/CD pipeline
- Когда не нужен SSL

---

## 2. `docker-compose-v2.yml` (production на сервере)

**Назначение:** Production развертывание на сервере с SSL

**Особенности:**
- ✅ Version 2 (для совместимости с сервером)
- ✅ **HTTPS с SSL сертификатами**
- ✅ Let's Encrypt сертификаты (geos.icc.ru)
- ✅ Автоматический рестарт
- ✅ Использует порты из .env

**Пути к сертификатам:**
```
/etc/letsencrypt/live/geos.icc.ru/fullchain.pem
/etc/letsencrypt/live/geos.icc.ru/privkey.pem
```

**Использование:**
```bash
# На сервере
docker-compose -f docker-compose-v2.yml up -d --build

# Проверка
docker-compose -f docker-compose-v2.yml ps
docker-compose -f docker-compose-v2.yml logs -f app

# Перезапуск после обновления сертификатов
docker-compose -f docker-compose-v2.yml restart app
```

**Переменные окружения (.env):**
```bash
# Порты (одинаковые с локальной разработкой)
NODE_LOCAL_PORT=6868       # Внешний порт на сервере
NODE_DOCKER_PORT=8080      # Внутренний порт приложения

# SSL включен автоматически
SSL_ENABLED=true
SSL_KEY_PATH=/certs/privkey.pem
SSL_CERT_PATH=/certs/fullchain.pem
```

**Когда использовать:**
- Production сервер
- Когда нужен HTTPS
- Когда есть Let's Encrypt сертификаты
- Для публичного доступа

---

## 3. `docker-compose-db.yml` (только БД)

**Назначение:** Запуск только PostgreSQL для локальной разработки

**Особенности:**
- ✅ Только база данных
- ✅ Без приложения
- ✅ Для локальной разработки Python кода

**Использование:**
```bash
# Запуск только БД
docker-compose -f docker-compose-db.yml up -d

# Приложение запускается локально
python main.py
```

**Когда использовать:**
- Разработка без Docker для приложения
- Отладка Python кода в IDE
- Быстрое тестирование изменений

---

## Сравнение конфигураций

| Параметр | docker-compose.yml | docker-compose-v2.yml | docker-compose-db.yml |
|----------|-------------------|----------------------|---------------------|
| **Version** | 3.8 | 2 | 3.8 |
| **SSL/HTTPS** | ❌ HTTP | ✅ HTTPS | N/A |
| **Сертификаты** | Нет | Let's Encrypt | N/A |
| **Healthcheck** | ✅ Да | ❌ Нет | ❌ Нет |
| **Порт по умолчанию** | 6868 (внешний) | 6868 (внешний) | 5432 |
| **Назначение** | Разработка | Production | Только БД |
| **wait-for-it** | ✅ Да | ❌ Нет (sleep) | N/A |
| **Приложение** | ✅ Да | ✅ Да | ❌ Нет |

---

## Быстрый старт

### Локальная разработка (HTTP):
```bash
docker-compose up -d --build
# Откройте: http://localhost:6868
```

### Production на сервере (HTTPS):
```bash
docker-compose -f docker-compose-v2.yml up -d --build
# Откройте: https://geos.icc.ru:6868
```

### Только БД для разработки:
```bash
docker-compose -f docker-compose-db.yml up -d
python main.py
# Откройте: http://localhost:8080
```

---

## SSL на production сервере

### Требования:
1. ✅ Let's Encrypt сертификаты установлены
2. ✅ Сертификаты доступны по пути: `/etc/letsencrypt/live/geos.icc.ru/`
3. ✅ Порт 6868 открыт
4. ✅ DNS настроен на сервер

### Проверка сертификатов:
```bash
# На сервере
ls -la /etc/letsencrypt/live/geos.icc.ru/
# Должны быть: fullchain.pem и privkey.pem

# Проверить срок действия
openssl x509 -in /etc/letsencrypt/live/geos.icc.ru/fullchain.pem -noout -dates
```

### Обновление сертификатов:
```bash
# Обновить Let's Encrypt сертификаты
sudo certbot renew

# Перезапустить приложение
docker-compose -f docker-compose-v2.yml restart app
```

---

## Переменные окружения

### Обязательные для всех:
```bash
# Database
POSTGRESDB_USER=postgres
POSTGRESDB_ROOT_PASSWORD=your_password
POSTGRESDB_DATABASE=rec_system
POSTGRESDB_LOCAL_PORT=5432
POSTGRESDB_DOCKER_PORT=5432
```

### Для docker-compose.yml (локально):
```bash
NODE_LOCAL_PORT=8080
NODE_DOCKER_PORT=8080
DEBUG=true
ENABLE_CRON=false
```

### Для docker-compose-v2.yml (production):
```bash
NODE_LOCAL_PORT=8080     # Внешний порт на сервере
NODE_DOCKER_PORT=8080    # Внутренний порт
DEBUG=false
ENABLE_CRON=true
SSL_ENABLED=true
SSL_KEY_PATH=/certs/privkey.pem
SSL_CERT_PATH=/certs/fullchain.pem
```

---

## Troubleshooting

### Проблема: SSL сертификаты не найдены

```bash
# Проверьте пути
docker-compose -f docker-compose-v2.yml run app ls -la /certs/

# Проверьте права
ls -la /etc/letsencrypt/live/geos.icc.ru/

# Убедитесь что Docker может читать файлы
sudo chmod 644 /etc/letsencrypt/live/geos.icc.ru/*.pem
sudo chmod 755 /etc/letsencrypt/live/geos.icc.ru/
```

### Проблема: Порт 6868 занят

```bash
# Проверьте что использует порт
sudo lsof -i :6868

# Остановите конфликтующий сервис
sudo systemctl stop <service_name>

# Или измените порт в .env
NODE_LOCAL_PORT=6869
```

### Проблема: База данных не готова

```bash
# Для docker-compose.yml используется wait-for-it.sh
# Для docker-compose-v2.yml увеличьте sleep если нужно:
# В файле измените: sleep 10 -> sleep 20
```

---

## Миграция между конфигурациями

### Из локальной в production:

```bash
# 1. Остановить локальную версию
docker-compose down

# 2. Создать backup БД (опционально)
docker-compose exec postgresdb pg_dump -U postgres rec_system > backup.sql

# 3. Запустить production версию на сервере
docker-compose -f docker-compose-v2.yml up -d --build

# 4. Восстановить БД если нужно
cat backup.sql | docker-compose -f docker-compose-v2.yml exec -T postgresdb psql -U postgres rec_system
```

---

## Рекомендации

### Для разработки:
- ✅ Используйте `docker-compose.yml`
- ✅ Включите DEBUG=true
- ✅ Отключите CRON (ENABLE_CRON=false)
- ✅ Используйте HTTP (быстрее)

### Для production:
- ✅ Используйте `docker-compose-v2.yml`
- ✅ Включите SSL (SSL_ENABLED=true)
- ✅ Включите CRON (ENABLE_CRON=true)
- ✅ Отключите DEBUG (DEBUG=false)
- ✅ Убедитесь что порт 6868 открыт в firewall
- ✅ Регулярно обновляйте сертификаты
- ✅ Настройте мониторинг логов

---

## Автоматизация на production

### Systemd service (опционально):

Создайте `/etc/systemd/system/rec-system.service`:

```ini
[Unit]
Description=Recommendation System
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/path/to/rec-system-services-geo-python
ExecStart=/usr/local/bin/docker-compose -f docker-compose-v2.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose-v2.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

Активация:
```bash
sudo systemctl daemon-reload
sudo systemctl enable rec-system
sudo systemctl start rec-system
```

---

**Последнее обновление:** 9 октября 2025

