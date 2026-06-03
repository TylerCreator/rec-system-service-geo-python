## Отчёт о выполнении пунктов 3, 4, 5

---

### Пункт 3: Проверить как записываются логи о вызове сервисов — ✅ Выполнен

Логи хранятся в таблице `Calls` (33 260 записей). Каждая запись содержит:

| Поле | Что хранит | Пример (mid=399, mapcombine) |
|---|---|---|
| `id` | ID вызова | 41803 |
| `mid` | ID сервиса | 399 |
| `owner` | ID пользователя | 50f7a1d80d58140037000006 |
| `status` | Статус | TASK_SUCCEEDED |
| `input` | Входные данные (JSON) | `{"tables":[],"new_table":{"dataset_id":"3086",...}}` |
| `result` | Выходные данные (JSON) | `{"tables":[{"dataset_id":"3086",...}]}` |
| `start_time` | Время начала | 2026-01-30T17:14:52 |

Для mid=399 (комбайн/mapcombine): вход принимает `tables` (список уже добавленных таблиц) и `new_table` (новая таблица с `dataset_id`), выход — обновлённый `tables`.

---

### Пункт 4: Дописать метод вычленения композиций — ✅ Выполнен

#### Таблица `Compositions` (верхняя картинка — полный DAG сервисов и таблиц)

3 123 записей. Каждая хранит `nodes` (узлы: таблицы + сервисы) и `links` (связи между ними).

Распределение по размеру:

| Нод | Кол-во композиций |
|---|---|
| 2 | 3 109 |
| 3 | 7 |
| 4 | 1 |
| 5 | 2 |
| 6 | 1 |
| 7 | 1 |
| 8 | 2 |

Пример реальной 3-нодной композиции (цепочка `309→309→309`):
```
id: "32086_32089_32091"
nodes: [
  {"id": 32086, "mid": 309, "owner": "54d2c6ba..."},
  {"id": 32089, "mid": 309, "owner": "50f7a1d8..."},
  {"id": 32091, "mid": 309, "owner": "50f7a1d8..."}
]
links: [
  {"source": 32086, "target": 32089, "fields": ["static_map:publish"]},
  {"source": 32089, "target": 32091, "fields": ["static_map:publish"]}
]
```

Самая длинная цепочка — 8 нод: `309→309→309→309→309→309→309→403`.

#### Таблица `TableCompositions` (нижняя картинка — таблица → таблица)

3 123 записей. Каждая хранит:
- `table_ids` — какие таблицы участвуют
- `service_mids` — через какие сервисы
- `join_steps` — где происходит слияние веток (is_join=true если есть upstream + table)

Статистика:

| Метрика | Значение |
|---|---|
| Всего | 3 123 |
| С таблицами | 3 099 |
| С 2+ таблицами | 1 |
| С 2+ сервисами | 17 |
| С is_join=true | 5 |

Пример multi-table TableComposition (соответствует картинке: table_1 + table_2 → комбайн):
```
id: "41800"
table_ids: [1003284, 1002118]
service_mids: [399]
join_steps: [{
  "target_service_mid": 399,
  "table_inputs": [
    {"table_id": 1003284, "fields": "1003284:tables"},
    {"table_id": 1002118, "fields": "1002118:new_table"}
  ],
  "is_join": true
}]
```

---

### Пункт 5: Рекомендательная модель для подстановки новой таблицы — ✅ Выполнен

Эндпоинт: `POST /table-compositions/recommend/substitute-table`

Запрос:
```json
{
  "upstream_service_id": 399,
  "new_table_id": 1003086,
  "n": 5
}
```

Ответ (реальный, из текущей БД):
```json
{
  "candidates": [
    {
      "service_chain": [399],
      "score": 3,
      "evidence_count": 3,
      "first_service": {
        "service_id": 399,
        "table_input_params": ["tables", "new_table"]
      },
      "join_service": {
        "service_id": 399,
        "artifact_input_params": ["map", "new_layer_wms_link"]
      },
      "examples": [
        {"composition_id": "41797", "table_inputs": [{"table_id": 1003284}]},
        {"composition_id": "41800", "table_inputs": [{"table_id": 1003284}, {"table_id": 1002118}]},
        {"composition_id": "41803", "table_inputs": [{"table_id": 1003086}]}
      ]
    },
    {
      "service_chain": [1000225],
      "score": 1
    },
    {
      "service_chain": [1003093],
      "score": 1
    }
  ],
  "raw_patterns_found": 3,
  "table_compositions_used": 3123
}
```

Модель рекомендует:
1. **mid=399** (комбайн) — score=3, подтверждён 3 реальными композициями. Принимает `tables` + `new_table` на вход.
2. **mid=1000225** — score=1, альтернативный сервис обработки таблицы после комбайна.
3. **mid=1003093** — score=1, ещё один альтернативный сервис.

Это соответствует схеме: "подставь новую таблицу (table_2) в комбайн (wps_1, mid=399)".

---

### Что было исправлено в процессе

1. **Fingerprint по ВСЕМ ключам** — раньше сканировались только configured keys из Services (WPS-параметры `map`, `new_layer_wms_link`), а реальные лог-поля (`tables`, `new_table`, `source`) игнорировались → не строились цепочки.
2. **`in_and_out_settings.json`** — добавлены реальные параметры mid=399 и mid=309.
3. **Multi-input merge** — при нескольких входах композиция теперь мерджится, а не перезаписывается.
4. **`API_TIMEOUT=300`** — увеличен с 90 для стабильной загрузки из CRIS.

---

### Как воспроизвести

```bash
# 1. Запуск
docker compose up -d --build

# 2. Загрузка Services + Datasets
curl --max-time 600 http://localhost:6868/services/update
curl --max-time 600 http://localhost:6868/datasets/update

# 3. Очистка + восстановление композиций
docker compose exec -T postgresdb psql -U postgres -d compositions -c \
  "DELETE FROM \"Compositions\"; DELETE FROM \"TableCompositions\";"
curl --max-time 300 http://localhost:6868/compositions/recoverNew

# 4. Проверка
docker compose exec -T postgresdb psql -U postgres -d compositions -c \
  "SELECT json_array_length(nodes::json) AS n, COUNT(*) FROM \"Compositions\" GROUP BY 1 ORDER BY 1;"

# 5. Рекомендация
curl -X POST http://localhost:6868/table-compositions/recommend/substitute-table \
  -H "Content-Type: application/json" \
  -d '{"upstream_service_id":399,"new_table_id":1003086,"n":5}'
```

---

### Список изменённых файлов

**Добавлено:**
- `app/services/compositions/table_compositions.py`
- `app/services/table_compositions_service.py`
- `app/routers/table_compositions.py`

**Изменено:**
- `app/models/models.py` — модель `TableComposition`
- `app/services/compositions/recovery.py` — multi-input merge + fingerprint all keys + persist Compositions/TableCompositions
- `app/services/compositions/repository.py` — CRUD TableCompositions
- `app/services/compositions/service_map.py` — fallback to file-defined services
- `app/services/compositions/__init__.py` — exports
- `app/services/compositions/helpers.py` — normalize_dataset_id для строковых чисел
- `app/static/in_and_out_settings.json` — реальные параметры mid=399, mid=309
- `app/routers/services.py` — `GET /services/update`
- `main.py` — подключение роутера
- `app/services/__init__.py` — экспорт сервиса
- `.env` — `API_TIMEOUT=300`
