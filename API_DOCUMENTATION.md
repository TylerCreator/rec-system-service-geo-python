# API Documentation - Service Parameters Endpoint

## Получение возможных параметров сервиса

### Endpoint
```
GET /services/parameters/:serviceId
```

### Описание
Возвращает возможные параметры, с которыми мог быть вызван указанный сервис, на основе исторических данных о вызовах.

### Параметры

#### Path Parameters
- `serviceId` (number, required) - ID сервиса

#### Query Parameters
- `user` (string, optional) - Фильтровать по конкретному пользователю
- `limit` (number, optional, default: 100) - Ограничить количество результатов
- `unique` (boolean, optional, default: true) - Вернуть только уникальные комбинации параметров

### Примеры запросов

#### Базовый запрос
```bash
GET /services/parameters/308
```

#### С фильтрацией по пользователю
```bash
GET /services/parameters/308?user=50f7a1d80d58140037000006
```

#### С ограничением результатов
```bash
GET /services/parameters/308?limit=50&unique=false
```

### Структура ответа

```json
{
  "service": {
    "id": 308,
    "name": "Сервис экспорта данных",
    "description": "Экспорт географических данных",
    "type": "export"
  },
  "parameters": [
    {
      "callId": 7208,
      "owner": "50f7a1d80d58140037000006",
      "timestamp": "2020-03-16T12:24:15.000Z",
      "status": "TASK_SUCCEEDED",
      "parameters": {
        "theme": {
          "dataset_id": "061672d8-eee3-4e68-aeea-39d868814085",
          "FilterValues": {},
          "groupdata": {}
        }
      }
    }
  ],
  "analysis": {
    "parameterNames": {
      "theme": 5
    },
    "parameterTypes": {
      "theme": {
        "object": 5
      }
    },
    "mostCommonValues": {
      "theme.dataset_id": {
        "061672d8-eee3-4e68-aeea-39d868814085": 3,
        "another-dataset-id": 2
      }
    },
    "totalUniqueCombinations": 5
  },
  "schema": [
    {
      "fieldname": "theme",
      "type": "theme_select",
      "widget": {
        "name": "theme_select"
      }
    }
  ],
  "totalCalls": 10,
  "returnedParameters": 5,
  "filters": {
    "user": null,
    "limit": 100,
    "unique": true
  }
}
```

### Поля ответа

#### service
- `id` - ID сервиса
- `name` - Название сервиса
- `description` - Описание сервиса
- `type` - Тип сервиса

#### parameters
Массив объектов с историческими вызовами:
- `callId` - ID вызова
- `owner` - Владелец/пользователь, сделавший вызов
- `timestamp` - Время вызова
- `status` - Статус выполнения вызова
- `parameters` - Параметры, с которыми был вызван сервис

#### analysis
Статистический анализ параметров:
- `parameterNames` - Частота использования имен параметров
- `parameterTypes` - Типы данных для каждого параметра
- `mostCommonValues` - Наиболее часто используемые значения
- `totalUniqueCombinations` - Общее количество уникальных комбинаций

#### schema
Схема параметров сервиса из его описания (если доступна)

#### Метаданные
- `totalCalls` - Общее количество найденных вызовов
- `returnedParameters` - Количество возвращенных параметров
- `filters` - Примененные фильтры

### Коды ошибок

#### 400 Bad Request
```json
{
  "error": "Invalid service ID. Service ID must be a number.",
  "serviceId": "invalid"
}
```

#### 404 Not Found
```json
{
  "error": "Service not found",
  "serviceId": 999
}
```

#### 500 Internal Server Error
```json
{
  "error": "Internal server error",
  "message": "Database connection failed"
}
```

### Примеры использования

#### cURL
```bash
# Получить параметры сервиса 308
curl -X GET "http://localhost:8080/services/parameters/308"

# Получить параметры для конкретного пользователя
curl -X GET "http://localhost:8080/services/parameters/308?user=50f7a1d80d58140037000006"

# Получить все вызовы (не только уникальные)
curl -X GET "http://localhost:8080/services/parameters/308?unique=false&limit=200"
```

#### JavaScript (fetch)
```javascript
// Базовый запрос
const response = await fetch('/services/parameters/308');
const data = await response.json();

// С параметрами
const response = await fetch('/services/parameters/308?user=50f7a1d80d58140037000006&limit=50');
const data = await response.json();

console.log('Найдено параметров:', data.returnedParameters);
console.log('Анализ параметров:', data.analysis);
```

### Особенности работы

1. **Уникальность**: По умолчанию возвращаются только уникальные комбинации параметров
2. **Сортировка**: Результаты сортируются по времени (новые сначала)
3. **Обработка ошибок**: Некорректные JSON-параметры пропускаются с предупреждением
4. **Производительность**: Для больших объемов данных используется ограничение и пагинация
5. **Анализ**: Автоматический анализ типов данных и популярных значений

