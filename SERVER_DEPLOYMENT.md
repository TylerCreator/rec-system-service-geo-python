# 🚀 Развертывание на production сервере

## Быстрая инструкция для geos.icc.ru

---

## ✅ Предварительные требования

1. ✅ Сервер с Docker и Docker Compose
2. ✅ Let's Encrypt сертификаты установлены для geos.icc.ru
3. ✅ Порт 6868 открыт
4. ✅ DNS настроен на IP сервера

---

## 📦 Шаг 1: Подготовка сервера

### Проверка сертификатов:
```bash
# Проверить что сертификаты на месте
ls -la /etc/letsencrypt/live/geos.icc.ru/
# Должны быть: fullchain.pem и privkey.pem

# Проверить срок действия
sudo openssl x509 -in /etc/letsencrypt/live/geos.icc.ru/fullchain.pem -noout -dates
```

### Настройка прав доступа:
```bash
# Docker должен иметь доступ к сертификатам
sudo chmod 755 /etc/letsencrypt/live/
sudo chmod 755 /etc/letsencrypt/live/geos.icc.ru/
sudo chmod 644 /etc/letsencrypt/live/geos.icc.ru/*.pem
```

---

## 📥 Шаг 2: Загрузка кода

```bash
# Перейти в рабочую директорию
cd /path/to/deployment

# Клонировать/скопировать проект
git clone <repository> rec-system-services-geo-python
cd rec-system-services-geo-python

# Или обновить существующий
git pull origin main
```

---

## ⚙️ Шаг 3: Настройка переменных окружения

```bash
# Создать .env файл из примера
cp env.example .env

# Отредактировать .env
nano .env
```

**Важные настройки для production:**

```bash
# Database
POSTGRESDB_USER=postgres
POSTGRESDB_ROOT_PASSWORD=STRONG_PASSWORD_HERE  # ⚠️ Изменить!
POSTGRESDB_DATABASE=rec_system
POSTGRESDB_LOCAL_PORT=5432
POSTGRESDB_DOCKER_PORT=5432

# Application - порты (одинаковые с локальной разработкой)
NODE_LOCAL_PORT=6868         # ⚠️ Внешний порт
NODE_DOCKER_PORT=8080        # Внутренний порт приложения

# Production settings
DEBUG=false                  # ⚠️ Отключить отладку
ENABLE_CRON=true            # Включить автообновления
PORT=8080

# SSL - автоматически настроены в docker-compose-v2.yml
SSL_ENABLED=true
SSL_KEY_PATH=/certs/privkey.pem
SSL_CERT_PATH=/certs/fullchain.pem

# External API
CRIS_BASE_URL=http://cris.icc.ru
API_TIMEOUT=90
```

---

## 🐳 Шаг 4: Запуск приложения

### Первый запуск:

```bash
# Собрать и запустить
docker-compose -f docker-compose-v2.yml up -d --build

# Проверить статус
docker-compose -f docker-compose-v2.yml ps

# Посмотреть логи
docker-compose -f docker-compose-v2.yml logs -f app
```

### Проверка работы:

```bash
# Проверить что контейнеры запущены
docker-compose -f docker-compose-v2.yml ps
# Должны быть: app (up), postgresdb (up)

# Проверить логи приложения
docker-compose -f docker-compose-v2.yml logs app | tail -50

# Проверить что порт 6868 слушается
sudo netstat -tlnp | grep :6868

# Проверить доступность API
curl -k https://localhost:6868/
curl -k https://localhost:6868/docs
```

### Проверка из браузера:

Откройте: **https://geos.icc.ru:6868/**

Должны увидеть:
```json
{
  "message": "Service Recommendation System API",
  "status": "running",
  "version": "2.0.0"
}
```

Документация: **https://geos.icc.ru:6868/docs**

---

## 🔄 Обновление приложения

```bash
# Перейти в директорию проекта
cd /path/to/rec-system-services-geo-python

# Получить последние изменения
git pull origin main

# Пересобрать и перезапустить
docker-compose -f docker-compose-v2.yml up -d --build

# Проверить логи
docker-compose -f docker-compose-v2.yml logs -f app
```

---

## 📊 Управление

### Основные команды:

```bash
# Запуск
docker-compose -f docker-compose-v2.yml up -d

# Остановка
docker-compose -f docker-compose-v2.yml down

# Перезапуск
docker-compose -f docker-compose-v2.yml restart

# Перезапуск только приложения
docker-compose -f docker-compose-v2.yml restart app

# Логи приложения
docker-compose -f docker-compose-v2.yml logs -f app

# Логи базы данных
docker-compose -f docker-compose-v2.yml logs -f postgresdb

# Статус контейнеров
docker-compose -f docker-compose-v2.yml ps

# Войти в контейнер приложения
docker-compose -f docker-compose-v2.yml exec app bash

# Войти в PostgreSQL
docker-compose -f docker-compose-v2.yml exec postgresdb psql -U postgres -d rec_system
```

---

## 🔒 Обновление SSL сертификатов

Let's Encrypt сертификаты нужно обновлять каждые 90 дней.

### Ручное обновление:

```bash
# Обновить сертификаты
sudo certbot renew

# Проверить новые сертификаты
sudo openssl x509 -in /etc/letsencrypt/live/geos.icc.ru/fullchain.pem -noout -dates

# Перезапустить приложение чтобы применить новые сертификаты
docker-compose -f docker-compose-v2.yml restart app
```

### Автоматическое обновление:

Создайте cron задачу:

```bash
sudo crontab -e
```

Добавьте:
```bash
# Обновление Let's Encrypt сертификатов каждый день в 3:00
0 3 * * * certbot renew --quiet && docker-compose -f /path/to/rec-system-services-geo-python/docker-compose-v2.yml restart app
```

---

## 💾 Backup базы данных

### Создать backup:

```bash
# Backup с датой в имени
docker-compose -f docker-compose-v2.yml exec postgresdb pg_dump -U postgres rec_system > backup_$(date +%Y%m%d_%H%M%S).sql

# Или в gzip
docker-compose -f docker-compose-v2.yml exec postgresdb pg_dump -U postgres rec_system | gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz
```

### Восстановить из backup:

```bash
# Из SQL файла
cat backup.sql | docker-compose -f docker-compose-v2.yml exec -T postgresdb psql -U postgres rec_system

# Из gzip
gunzip -c backup.sql.gz | docker-compose -f docker-compose-v2.yml exec -T postgresdb psql -U postgres rec_system
```

### Автоматический backup (cron):

```bash
sudo crontab -e
```

Добавьте:
```bash
# Backup каждый день в 2:00
0 2 * * * cd /path/to/rec-system-services-geo-python && docker-compose -f docker-compose-v2.yml exec postgresdb pg_dump -U postgres rec_system | gzip > /backups/rec_system_$(date +\%Y\%m\%d).sql.gz

# Удаление старых backup (старше 30 дней)
0 3 * * * find /backups -name "rec_system_*.sql.gz" -mtime +30 -delete
```

---

## 🔍 Мониторинг и troubleshooting

### Проверка ресурсов:

```bash
# Использование ресурсов контейнерами
docker stats

# Размер контейнеров и образов
docker system df

# Логи за последний час
docker-compose -f docker-compose-v2.yml logs --since 1h app
```

### Частые проблемы:

#### 1. Контейнер не запускается

```bash
# Проверить логи
docker-compose -f docker-compose-v2.yml logs app

# Проверить что порты свободны
sudo netstat -tlnp | grep :443
sudo netstat -tlnp | grep :8080

# Проверить конфигурацию
docker-compose -f docker-compose-v2.yml config
```

#### 2. SSL ошибки

```bash
# Проверить сертификаты в контейнере
docker-compose -f docker-compose-v2.yml exec app ls -la /certs/

# Проверить что приложение видит сертификаты
docker-compose -f docker-compose-v2.yml exec app cat /certs/fullchain.pem | head -5
```

#### 3. База данных недоступна

```bash
# Проверить статус PostgreSQL
docker-compose -f docker-compose-v2.yml ps postgresdb

# Перезапустить БД
docker-compose -f docker-compose-v2.yml restart postgresdb

# Проверить логи БД
docker-compose -f docker-compose-v2.yml logs postgresdb
```

#### 4. Медленная работа

```bash
# Проверить использование памяти
docker stats --no-stream

# Проверить логи на ошибки
docker-compose -f docker-compose-v2.yml logs app | grep -i error

# Перезапустить приложение
docker-compose -f docker-compose-v2.yml restart app
```

---

## 📈 Первоначальная загрузка данных

После первого запуска база данных будет пустой:

```bash
# Вариант 1: Через API (займет время!)
curl -k https://geos.icc.ru/update/full

# Вариант 2: Внутри контейнера
docker-compose -f docker-compose-v2.yml exec app python -c "
from app.services.update_service import run_full_update
import asyncio
asyncio.run(run_full_update())
"

# Проверить прогресс в логах
docker-compose -f docker-compose-v2.yml logs -f app
```

---

## 🔐 Безопасность

### Рекомендации:

1. ✅ Используйте сильный пароль БД в `.env`
2. ✅ Регулярно обновляйте Docker образы
3. ✅ Настройте firewall (ufw/iptables)
4. ✅ Регулярно обновляйте SSL сертификаты
5. ✅ Делайте backup базы данных
6. ✅ Мониторьте логи на подозрительную активность
7. ✅ Ограничьте доступ к серверу (SSH keys only)

### Firewall (пример с ufw):

```bash
# Разрешить SSH
sudo ufw allow 22/tcp

# Разрешить приложение
sudo ufw allow 6868/tcp

# Включить firewall
sudo ufw enable

# Проверить статус
sudo ufw status
```

---

## 📱 Systemd service (опционально)

Создайте systemd service для автозапуска:

```bash
sudo nano /etc/systemd/system/rec-system.service
```

Содержимое:
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
sudo systemctl status rec-system
```

---

## ✅ Checklist развертывания

- [ ] Сервер подготовлен (Docker установлен)
- [ ] SSL сертификаты на месте и актуальны
- [ ] Порт 6868 открыт в firewall
- [ ] DNS настроен на IP сервера
- [ ] `.env` файл создан и настроен
- [ ] Приложение запущено: `docker-compose -f docker-compose-v2.yml up -d --build`
- [ ] API доступен: https://geos.icc.ru:6868/
- [ ] Swagger UI доступен: https://geos.icc.ru:6868/docs
- [ ] Логи проверены на ошибки
- [ ] Данные загружены: `/update/full`
- [ ] Backup настроен (cron)
- [ ] SSL auto-renewal настроен (cron)
- [ ] Мониторинг настроен

---

## 📞 Поддержка

При проблемах:

1. Проверьте логи: `docker-compose -f docker-compose-v2.yml logs -f app`
2. Проверьте статус: `docker-compose -f docker-compose-v2.yml ps`
3. Прочитайте документацию: `DOCKER_COMPOSE_INFO.md`
4. Проверьте troubleshooting секцию выше

---

**Последнее обновление:** 9 октября 2025  
**Версия:** 2.0.0 (Python/FastAPI)

