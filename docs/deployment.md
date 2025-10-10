# 🚀 Развертывание на Production

Инструкция по развертыванию Service Recommendation System на production сервере.

## Подготовка сервера

### Требования

- Ubuntu 20.04+ / Debian 11+
- Docker и Docker Compose
- SSL сертификаты (Let's Encrypt рекомендуется)
- Открытые порты: 80, 443, 6868

### Установка Docker

```bash
# Обновить систему
sudo apt update && sudo apt upgrade -y

# Установить Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Установить Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Добавить пользователя в группу docker
sudo usermod -aG docker $USER
newgrp docker

# Проверить установку
docker --version
docker-compose --version
```

## Развертывание с Docker Compose

### Вариант 1: HTTP (без SSL)

Для локальных серверов или development окружения.

```bash
# 1. Клонировать репозиторий
cd /opt
sudo git clone <repository-url> rec-system
cd rec-system

# 2. Настроить окружение
sudo cp env.example .env
sudo nano .env
```

Настройки для production:
```bash
# Порты
NODE_LOCAL_PORT=6868
POSTGRESDB_LOCAL_PORT=5431

# База данных
POSTGRESDB_USER=postgres
POSTGRESDB_ROOT_PASSWORD=root              # Для dev/test
# POSTGRESDB_ROOT_PASSWORD=<сильный_пароль>  # Для production
POSTGRESDB_DATABASE=compositions

# Приложение
PORT=8080
DEBUG=false
ENABLE_CRON=true

# API
CRIS_BASE_URL=http://cris.icc.ru
API_TIMEOUT=90
```

```bash
# 3. Запустить
sudo docker-compose up -d --build

# 4. Проверить
curl http://localhost:6868/
sudo docker-compose logs -f
```

### Вариант 2: HTTPS (с SSL)

Для production серверов с доменом.

#### Получение SSL сертификатов

```bash
# Установить certbot
sudo apt install certbot

# Получить сертификат (замените на свой домен)
sudo certbot certonly --standalone -d geos.icc.ru

# Сертификаты будут в:
# /etc/letsencrypt/live/geos.icc.ru/fullchain.pem
# /etc/letsencrypt/live/geos.icc.ru/privkey.pem
```

#### Настройка docker-compose-v2.yml

```bash
# Использовать версию с SSL
sudo nano docker-compose-v2.yml
```

Проверьте volumes для сертификатов:
```yaml
services:
  app:
    volumes:
      - /etc/letsencrypt/live/geos.icc.ru:/certs:ro
```

#### Настройка .env

```bash
sudo nano .env
```

Включите SSL:
```bash
# SSL
SSL_ENABLED=true
SSL_KEY_PATH=/certs/privkey.pem
SSL_CERT_PATH=/certs/fullchain.pem
```

#### Запуск

```bash
# Запустить с SSL конфигурацией
sudo docker-compose -f docker-compose-v2.yml up -d --build

# Проверить
curl https://geos.icc.ru:6868/
```

## Nginx Reverse Proxy (опционально)

Для работы на стандартных портах 80/443.

### Установка Nginx

```bash
sudo apt install nginx
```

### Конфигурация

```bash
sudo nano /etc/nginx/sites-available/rec-system
```

Содержимое:
```nginx
server {
    listen 80;
    server_name geos.icc.ru;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name geos.icc.ru;
    
    # SSL certificates
    ssl_certificate /etc/letsencrypt/live/geos.icc.ru/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/geos.icc.ru/privkey.pem;
    
    # SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # Proxy settings
    location / {
        proxy_pass http://localhost:6868;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 90s;
        proxy_send_timeout 90s;
        proxy_read_timeout 90s;
    }
    
    # API docs
    location /docs {
        proxy_pass http://localhost:6868/docs;
    }
    
    location /redoc {
        proxy_pass http://localhost:6868/redoc;
    }
}
```

### Активация

```bash
# Создать симлинк
sudo ln -s /etc/nginx/sites-available/rec-system /etc/nginx/sites-enabled/

# Проверить конфигурацию
sudo nginx -t

# Перезагрузить Nginx
sudo systemctl reload nginx

# Включить автозапуск
sudo systemctl enable nginx
```

Теперь API доступен на:
- https://geos.icc.ru/
- https://geos.icc.ru/docs

## Systemd Service (альтернатива Docker)

Для запуска без Docker как systemd сервис.

### Создание сервиса

```bash
sudo nano /etc/systemd/system/rec-system.service
```

Содержимое:
```ini
[Unit]
Description=Service Recommendation System
After=network.target postgresql.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/rec-system
Environment="PATH=/opt/rec-system/venv/bin"
ExecStart=/opt/rec-system/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Запуск

```bash
# Перезагрузить systemd
sudo systemctl daemon-reload

# Запустить сервис
sudo systemctl start rec-system

# Включить автозапуск
sudo systemctl enable rec-system

# Проверить статус
sudo systemctl status rec-system

# Логи
sudo journalctl -u rec-system -f
```

## Мониторинг

### Проверка здоровья

```bash
# Health check
curl http://localhost:6868/

# Логи Docker
sudo docker-compose logs -f app

# Статус контейнеров
sudo docker-compose ps

# Использование ресурсов
docker stats
```

### Логирование

Логи приложения находятся в:
- Docker: `docker-compose logs app`
- Systemd: `journalctl -u rec-system`
- Файл: `server.log` (если настроено)

### Мониторинг производительности

```bash
# Использование ресурсов контейнерами
docker stats --no-stream

# Использование диска
df -h
du -sh app/static/*

# Проверка БД
docker-compose exec postgresdb psql -U postgres -d rec_system -c "SELECT count(*) FROM \"Calls\";"
```

## Обновление

### Обновление кода

```bash
cd /opt/rec-system

# Сохранить данные
sudo docker-compose exec postgresdb pg_dump -U postgres rec_system > backup.sql

# Обновить код
sudo git pull

# Пересобрать и запустить
sudo docker-compose down
sudo docker-compose up -d --build

# Проверить логи
sudo docker-compose logs -f app
```

### Обновление зависимостей

```bash
# Обновить requirements.txt
sudo nano requirements.txt

# Пересобрать образ
sudo docker-compose build app

# Перезапустить
sudo docker-compose up -d app
```

## Backup и Recovery

### Backup базы данных

```bash
# Создать backup
docker-compose exec postgresdb pg_dump -U postgres rec_system > backup_$(date +%Y%m%d_%H%M%S).sql

# Backup с docker-compose
docker-compose exec -T postgresdb pg_dump -U postgres rec_system | gzip > backup.sql.gz
```

### Восстановление

```bash
# Восстановить из backup
docker-compose exec -T postgresdb psql -U postgres rec_system < backup.sql

# Из gzip
gunzip -c backup.sql.gz | docker-compose exec -T postgresdb psql -U postgres rec_system
```

### Backup файлов

```bash
# Backup статических файлов
tar -czf static_backup_$(date +%Y%m%d).tar.gz app/static/

# Восстановление
tar -xzf static_backup_20250110.tar.gz
```

## Автоматизация

### Cron для backup

```bash
sudo crontab -e
```

Добавить:
```bash
# Ежедневный backup в 3:00 AM
0 3 * * * cd /opt/rec-system && docker-compose exec -T postgresdb pg_dump -U postgres rec_system | gzip > /opt/backups/rec_system_$(date +\%Y\%m\%d).sql.gz

# Очистка старых backup (старше 30 дней)
0 4 * * * find /opt/backups -name "rec_system_*.sql.gz" -mtime +30 -delete
```

### Автообновление SSL

```bash
# Certbot автоматически добавляет cron для обновления
# Проверить:
sudo systemctl status certbot.timer

# Тест обновления
sudo certbot renew --dry-run

# После обновления перезапустить приложение
sudo crontab -e
```

Добавить после обновления сертификата:
```bash
# Перезапуск после обновления SSL (каждое 1-е число месяца)
0 5 1 * * cd /opt/rec-system && docker-compose restart app
```

## Безопасность

### Firewall

```bash
# Установить UFW
sudo apt install ufw

# Базовые правила
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Разрешить SSH
sudo ufw allow ssh

# Разрешить HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Разрешить порт приложения (если без Nginx)
sudo ufw allow 6868/tcp

# Включить
sudo ufw enable

# Статус
sudo ufw status
```

### Настройки безопасности

В `.env`:
```bash
# Отключить debug в production
DEBUG=false

# Использовать сильные пароли для production
# Development/Testing:
POSTGRESDB_ROOT_PASSWORD=root
DB_PASSWORD=root
# Production (замените на сильный пароль):
# POSTGRESDB_ROOT_PASSWORD=<сложный_пароль_минимум_32_символа>
# DB_PASSWORD=<сложный_пароль_минимум_32_символа>

# Ограничить доступ к БД
DB_HOST=postgresdb  # не localhost для изоляции в Docker
DB_NAME=compositions
```

### Rate Limiting (с Nginx)

Добавить в nginx конфиг:
```nginx
# В http блок
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

# В server блок
location / {
    limit_req zone=api_limit burst=20 nodelay;
    proxy_pass http://localhost:6868;
}
```

## Решение проблем

### Приложение не запускается

```bash
# Проверить логи
sudo docker-compose logs app

# Проверить переменные окружения
sudo docker-compose config

# Пересоздать контейнеры
sudo docker-compose down
sudo docker-compose up -d --build
```

### База данных недоступна

```bash
# Проверить статус БД
sudo docker-compose ps postgresdb

# Проверить логи
sudo docker-compose logs postgresdb

# Перезапустить БД
sudo docker-compose restart postgresdb
```

### Нет свободного места

```bash
# Проверить место
df -h

# Очистить Docker
docker system prune -a

# Очистить логи
sudo truncate -s 0 /var/lib/docker/containers/*/*-json.log
```

### Медленная работа

```bash
# Проверить ресурсы
docker stats

# Увеличить лимиты памяти в docker-compose.yml
services:
  app:
    mem_limit: 2g
  postgresdb:
    mem_limit: 2g
```

## Полезные команды

```bash
# Остановить все
sudo docker-compose down

# Пересобрать образы
sudo docker-compose build

# Запустить в фоне
sudo docker-compose up -d

# Перезапустить один сервис
sudo docker-compose restart app

# Выполнить команду в контейнере
sudo docker-compose exec app bash

# Подключиться к БД
sudo docker-compose exec postgresdb psql -U postgres -d rec_system

# Просмотр логов в реальном времени
sudo docker-compose logs -f --tail=100

# Очистить все (ОСТОРОЖНО: удаляет данные!)
sudo docker-compose down -v
```

