# 📡 API Documentation

Описание всех эндпоинтов Service Recommendation System API.

## Интерактивная документация

После запуска приложения доступна автоматически сгенерированная документация:

- **Swagger UI:** http://localhost:6868/docs
- **ReDoc:** http://localhost:6868/redoc

## Базовый URL

```
http://localhost:6868
```

Для production замените на ваш домен, например: `https://geos.icc.ru:6868`

## Эндпоинты

### Health Check

#### GET /

Проверка работоспособности API.

**Response:**
```json
{
  "message": "Service Recommendation System API",
  "status": "running"
}
```

---

### Services

#### GET /services/

Получить список всех сервисов.

**Query Parameters:**
- `skip` (int, optional): Количество записей для пропуска (пагинация)
- `limit` (int, optional): Максимальное количество записей

**Response:**
```json
[
  {
    "id": 309,
    "name": "Geoprocessing Service",
    "type": "geoprocessing",
    "description": "Service description",
    "number_of_calls": 1523
  }
]
```

#### GET /services/statistics

Получить статистику по сервисам.

**Response:**
```json
{
  "totalServices": 105,
  "totalCalls": 23428,
  "topServices": [
    {
      "id": 309,
      "name": "Service Name",
      "calls": 1523
    }
  ]
}
```

#### GET /services/popular

Получить топ популярных сервисов.

**Query Parameters:**
- `limit` (int, default: 10): Количество сервисов

**Response:**
```json
[
  {
    "id": 309,
    "name": "Popular Service",
    "number_of_calls": 1523,
    "type": "geoprocessing"
  }
]
```

#### GET /services/recomendation/{user_id}

Получить рекомендации для пользователя на основе KNN.

**Path Parameters:**
- `user_id` (string): ID пользователя

**Response:**
```json
{
  "prediction": {
    "50f7a1d80d58140037000006": [
      309,
      45,
      78
    ]
  }
}
```

---

### Calls

#### GET /calls/

Получить историю вызовов сервисов.

**Query Parameters:**
- `skip` (int, optional): Пропустить записей
- `limit` (int, optional): Максимум записей
- `user_id` (string, optional): Фильтр по пользователю
- `service_id` (int, optional): Фильтр по сервису

**Response:**
```json
[
  {
    "id": 27553,
    "mid": 399,
    "owner": "50f7a1d80d58140037000006",
    "status": "TASK_SUCCEEDED",
    "start_time": "2025-10-10T12:52:23",
    "end_time": "2025-10-10T12:52:25"
  }
]
```

#### GET /calls/dump-csv

Экспортировать вызовы в CSV файл.

**Response:**
```json
{
  "message": "CSV file created successfully",
  "file": "app/static/calls.csv",
  "records": 23428
}
```

Создает файл `app/static/calls.csv` с колонками:
- `id` - ID вызова
- `mid` - ID сервиса
- `owner` - ID пользователя
- `start_time` - время начала

---

### Compositions

#### GET /compositions/

Получить все композиции сервисов.

**Response:**
```json
[
  {
    "id": "27553_27556",
    "nodes": [
      {
        "id": "task/1",
        "taskId": 27553,
        "mid": 399,
        "service": "mapcombine"
      }
    ],
    "links": [
      {
        "source": "task/1",
        "target": "task/2",
        "value": ["map:map"]
      }
    ]
  }
]
```

#### GET /compositions/recover

Восстановить композиции из истории вызовов.

Анализирует все вызовы и строит граф композиций на основе связей между входами/выходами.

**Response:**
```json
{
  "success": true,
  "message": "Service composition recovery completed",
  "compositionsCount": 120,
  "usersCount": 989
}
```

#### GET /compositions/recoverNew

Восстановление композиций улучшенным алгоритмом.

Использует продвинутый алгоритм с отслеживанием датасетов. Сохраняет результат в `app/static/compositionsDAG.json`.

**Response:**
```json
{
  "success": true,
  "message": "Advanced composition recovery completed",
  "compositionsCount": 118,
  "servicesCount": 105,
  "datasetsCount": 45
}
```

#### GET /compositions/stats

Получить статистику по композициям.

**Response:**
```json
{
  "totalCompositions": 120,
  "avgNodesPerComposition": 2.5,
  "topServices": [
    {
      "mid": 399,
      "name": "mapcombine",
      "usageCount": 45
    }
  ]
}
```

---

### Datasets

#### GET /datasets/

Получить список датасетов.

**Query Parameters:**
- `skip` (int, optional)
- `limit` (int, optional)

**Response:**
```json
[
  {
    "id": 1,
    "guid": "dataset-guid-123"
  }
]
```

---

### Update

#### GET /update/statistics

Обновить статистику сервисов.

Загружает данные из CRIS API и обновляет:
- Количество вызовов сервисов
- Связи user-service
- Популярные сервисы

**Response:**
```json
{
  "success": true,
  "message": "Statistics updated successfully",
  "servicesUpdated": 105
}
```

#### GET /update/full

Полное обновление всех данных.

Выполняет:
1. Обновление статистики
2. Экспорт вызовов в CSV
3. Обучение KNN модели
4. Генерация рекомендаций

**⚠️ Внимание:** Операция может занять несколько минут!

**Response:**
```json
[
  "Статистика сервисов обновлена",
  "CSV файл создан",
  "KNN модель обучена",
  "Рекомендации сгенерированы"
]
```

---

## Коды ответов

### Успешные ответы

- `200 OK` - Запрос выполнен успешно
- `201 Created` - Ресурс создан

### Ошибки клиента

- `400 Bad Request` - Неверный формат запроса
- `404 Not Found` - Ресурс не найден
- `422 Unprocessable Entity` - Ошибка валидации данных

### Ошибки сервера

- `500 Internal Server Error` - Внутренняя ошибка сервера
- `503 Service Unavailable` - Сервис временно недоступен

## Примеры использования

### cURL

```bash
# Health check
curl http://localhost:6868/

# Получить статистику
curl http://localhost:6868/services/statistics

# Популярные сервисы
curl http://localhost:6868/services/popular?limit=5

# Рекомендации
curl http://localhost:6868/services/recomendation/50f7a1d80d58140037000006

# Обновить данные
curl http://localhost:6868/update/full

# Восстановить композиции
curl http://localhost:6868/compositions/recover
```

### Python

```python
import requests

BASE_URL = "http://localhost:6868"

# Получить сервисы
response = requests.get(f"{BASE_URL}/services/")
services = response.json()

# Рекомендации для пользователя
user_id = "50f7a1d80d58140037000006"
response = requests.get(f"{BASE_URL}/services/recomendation/{user_id}")
recommendations = response.json()

# Обновить статистику
response = requests.get(f"{BASE_URL}/update/statistics")
result = response.json()
```

### JavaScript (fetch)

```javascript
const BASE_URL = "http://localhost:6868";

// Получить популярные сервисы
fetch(`${BASE_URL}/services/popular?limit=10`)
  .then(res => res.json())
  .then(data => console.log(data));

// Рекомендации
const userId = "50f7a1d80d58140037000006";
fetch(`${BASE_URL}/services/recomendation/${userId}`)
  .then(res => res.json())
  .then(data => console.log(data.prediction));
```

## Ограничения

### Rate Limiting

По умолчанию нет ограничений на количество запросов. Для production рекомендуется настроить rate limiting через Nginx или middleware.

### Timeouts

- Стандартные запросы: 30 секунд
- `/update/full`: 90 секунд (настраивается через `API_TIMEOUT`)

### Размер данных

- Максимальный размер ответа: не ограничен
- Рекомендуется использовать пагинацию для больших списков

## CORS

По умолчанию CORS не настроен. Для работы с frontend добавьте домены в `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Аутентификация

В текущей версии аутентификация отсутствует. API открыт для всех запросов.

Для добавления аутентификации:
1. Установите `fastapi-users` или `python-jose`
2. Добавьте JWT токены
3. Защитите эндпоинты с помощью `Depends(get_current_user)`

## Версионирование

Текущая версия API: **v1**

В будущем планируется версионирование через префикс `/api/v1/`.

## WebSocket

WebSocket поддержка не реализована. Для real-time updates используйте polling или Server-Sent Events.

## Дополнительно

Для более подробной информации используйте интерактивную документацию:
- Swagger UI: http://localhost:6868/docs
- ReDoc: http://localhost:6868/redoc

