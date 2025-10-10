# 🏗️ Архитектура кода

Описание структуры кода после рефакторинга.

## Структура app/services/

```
app/services/
├── __init__.py                 # Экспорты модулей
├── calls_service.py            # Работа с вызовами сервисов
├── compositions_service.py     # Wrapper для compositions модулей
├── datasets_service.py         # Работа с датасетами  
├── services_service.py         # Управление сервисами и рекомендации
├── update_service.py           # Обновление данных
│
├── compositions/               # Модули для композиций (разбит большой файл)
│   ├── __init__.py
│   ├── builder.py             # Построение и нормализация композиций
│   ├── helpers.py             # Вспомогательные функции
│   ├── recovery.py            # Алгоритмы восстановления
│   ├── repository.py          # Операции с БД
│   └── service_map.py         # Построение карты связей сервисов
│
└── utils/                      # Общие утилиты (устранение дублирования)
    ├── __init__.py
    ├── constants.py           # Константы
    ├── parsers.py             # Парсинг данных
    └── validators.py          # Валидация данных
```

## Модули

### utils/ - Общие утилиты

#### constants.py
Хранит все константы приложения:
- `DATASET_ID_OFFSET` - смещение ID датасетов
- `WIDGET_*` - типы виджетов
- `TASK_*` - статусы задач

```python
from app.services.utils.constants import WIDGET_FILE, TASK_SUCCEEDED
```

#### parsers.py
Функции парсинга данных:
- `safe_json_parse()` - безопасный парсинг JSON
- `parse_datetime()` - парсинг дат
- `parse_service_params()` - парсинг параметров сервисов
- `to_string()` - конвертация в строку

```python
from app.services.utils.parsers import safe_json_parse, parse_datetime
```

#### validators.py
Функции валидации:
- `is_hashable()` - проверка хешируемости значения
- `categorize_params()` - категоризация параметров

```python
from app.services.utils.validators import is_hashable
```

### compositions/ - Модули композиций

Большой файл `compositions_service.py` (905 строк) разбит на 5 специализированных модулей:

#### service_map.py
Построение карты связей сервисов:
- `build_service_connection_map()` - карта входов/выходов сервисов
- `build_dataset_guid_map()` - маппинг GUID → ID датасетов

#### helpers.py
Вспомогательные функции:
- `normalize_dataset_id()` - нормализация ID датасета
- `add_task_link()` - добавление связи между задачами
- `create_composition_node()` - создание узла композиции

#### builder.py
Построение композиций:
- `build_composition_for_task()` - построение композиции для задачи
- `normalize_composition()` - нормализация (локальные ID, сортировка)

#### recovery.py
Алгоритмы восстановления:
- `recover()` - базовый алгоритм восстановления
- `recover_new()` - улучшенный алгоритм с датасетами
- Внутренние функции: `_is_successful_with_wms()`, `_process_task_inputs()`, `_register_task_outputs()`

#### repository.py
Операции с базой данных:
- `create_compositions()` - сохранение композиций
- `create_users()` - создание пользователей
- `fetch_all_compositions()` - получение всех композиций
- `get_composition_stats()` - статистика композиций

## Принципы рефакторинга

### 1. DRY (Don't Repeat Yourself)

**Было:** Функции `parse_datetime`, `to_string`, `safe_json_parse` дублировались в 3 файлах
**Стало:** Одна реализация в `utils/parsers.py`

### 2. Single Responsibility Principle

**Было:** `compositions_service.py` - 905 строк, множество ответственностей
**Стало:** 5 модулей, каждый отвечает за свою область

### 3. Модульность

Каждый модуль может быть:
- Легко протестирован отдельно
- Изменен без влияния на другие
- Переиспользован в других частях приложения

### 4. Обратная совместимость

`compositions_service.py` остался как wrapper, все импорты работают:

```python
# Старый код продолжает работать
from app.services.compositions_service import recover

# Новый код может использовать подмодули
from app.services.compositions.recovery import recover
```

## Зависимости модулей

```
compositions_service.py (wrapper)
    └── compositions/
        ├── recovery.py
        │   ├── → service_map.py
        │   ├── → builder.py
        │   ├── → helpers.py
        │   ├── → repository.py
        │   └── → utils/ (parsers, constants, validators)
        │
        ├── builder.py
        │   ├── → helpers.py
        │   └── → utils/
        │
        ├── helpers.py
        │   └── → utils/
        │
        ├── service_map.py
        │   └── → utils/
        │
        └── repository.py
            └── models/

calls_service.py
    └── utils/parsers.py

services_service.py
    └── utils/parsers.py
```

## Преимущества новой структуры

### ✅ Читабельность
- Файлы меньше 300 строк (легко читать)
- Понятные имена модулей
- Четкое разделение ответственности

### ✅ Поддерживаемость
- Легко найти нужную функцию
- Изменения локализованы
- Меньше конфликтов при merge

### ✅ Тестируемость
- Каждый модуль можно тестировать отдельно
- Четкие границы между модулями
- Легко мокировать зависимости

### ✅ Переиспользование
- Общие утилиты доступны везде
- Нет дублирования кода
- Единый источник правды для констант

## Сравнение

### Было (до рефакторинга)
```
app/services/
├── compositions_service.py    905 строк  ← Монолит!
├── calls_service.py           256 строк  ← Дублирование parse_datetime, to_string
├── services_service.py        702 строки ← Дублирование parse_datetime, to_string
├── datasets_service.py         92 строки
└── update_service.py          383 строки
```

**Проблемы:**
- Дублирование кода в 3 файлах
- Огромный файл композиций (905 строк)
- Сложно найти нужную функцию
- Много ответственностей в одном файле

### Стало (после рефакторинга)
```
app/services/
├── compositions_service.py     48 строк  ← Wrapper
├── calls_service.py           213 строк  ← Чище, без дублирования
├── services_service.py        665 строк  ← Чище, без дублирования
├── datasets_service.py         92 строки ← Без изменений
├── update_service.py          383 строки ← Без изменений
│
├── compositions/              ← Разбито на модули!
│   ├── __init__.py             26 строк
│   ├── builder.py             125 строк  ← Построение композиций
│   ├── helpers.py             100 строк  ← Вспомогательные функции
│   ├── recovery.py            334 строки ← Алгоритмы восстановления
│   ├── repository.py          146 строк  ← Работа с БД
│   └── service_map.py         120 строк  ← Карта связей
│
└── utils/                     ← Общие утилиты!
    ├── __init__.py              7 строк
    ├── constants.py            23 строки ← Все константы
    ├── parsers.py             100 строк  ← Парсинг (без дублирования)
    └── validators.py           55 строк  ← Валидация
```

**Улучшения:**
- ✅ Нет дублирования кода
- ✅ Все файлы < 350 строк
- ✅ Четкая структура и ответственности
- ✅ Легко находить и изменять код
- ✅ Обратная совместимость сохранена

## Использование

### Импорт композиций (старый способ)
```python
from app.services import compositions_service

# Все функции доступны
await compositions_service.recover(db)
```

### Импорт композиций (новый способ)
```python
from app.services.compositions import recovery, builder

# Более явно и читаемо
await recovery.recover(db)
composition = builder.normalize_composition(nodes, links)
```

### Импорт утилит
```python
from app.services.utils import parse_datetime, TASK_SUCCEEDED, is_hashable

# Используется везде, один источник правды
datetime_obj = parse_datetime("2025-10-10 12:52:23")
if status == TASK_SUCCEEDED and is_hashable(value):
    # ...
```

## Миграция существующего кода

Все старые импорты продолжают работать:

```python
# Это работает (старый код)
from app.services.compositions_service import recover

# Это тоже работает (новый код)  
from app.services.compositions.recovery import recover
```

Нет необходимости менять существующий код!

## Расширение функциональности

### Добавление новой функции в compositions

1. Определите к какому модулю она относится:
   - Работа с БД? → `repository.py`
   - Построение композиций? → `builder.py`
   - Восстановление? → `recovery.py`
   - Вспомогательная? → `helpers.py`

2. Добавьте функцию в соответствующий модуль

3. Экспортируйте в `__init__.py` если нужно

4. Опционально добавьте в `compositions_service.py` для обратной совместимости

### Добавление новой утилиты

1. Определите тип:
   - Парсинг? → `utils/parsers.py`
   - Валидация? → `utils/validators.py`
   - Константа? → `utils/constants.py`

2. Добавьте функцию/константу

3. Используйте везде где нужно

## Best Practices

### ✅ DO
- Используйте type hints
- Документируйте функции (docstrings)
- Храните константы в `utils/constants.py`
- Переиспользуйте функции из `utils/`

### ❌ DON'T
- Не дублируйте код
- Не создавайте файлы > 400 строк
- Не смешивайте разные ответственности
- Не нарушайте обратную совместимость

## Производительность

Рефакторинг **не влияет на производительность**:
- Импорты выполняются один раз при старте
- Функции остались теми же
- Добавлен только тонкий слой wrapper'ов

## Тестирование

Для тестирования отдельных модулей:

```python
# Тест парсинга
from app.services.utils.parsers import parse_datetime
assert parse_datetime("2025-10-10 12:00:00") is not None

# Тест построения композиций
from app.services.compositions.builder import normalize_composition
result = normalize_composition(nodes, links)
assert "id" in result
```

## Дальнейшие улучшения

Возможные направления:
1. Добавить unit тесты для каждого модуля
2. Создать базовый класс для сервисов (опционально)
3. Добавить type hints везде
4. Создать отдельный модуль для работы с внешним API
5. Вынести конфигурацию сервисов в отдельный модуль

