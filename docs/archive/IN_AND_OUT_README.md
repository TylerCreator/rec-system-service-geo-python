# in_and_out_settings.json - Расширение конфигурации сервисов

## Описание

Файл `in_and_out_settings.json` (расположен в `app/static/in_and_out_settings.json`) позволяет вручную расширять или переопределять конфигурацию входов и выходов сервисов, которая автоматически определяется из базы данных.

## Как работает

### 1. Автоматическое определение

Функция `build_service_connection_map()` автоматически анализирует:
- Параметры сервиса из БД (`service.params`)
- Выходные параметры (`service.output_params`)
- Типы виджетов (`file`, `file_save`, `theme_select`)

### 2. Расширение из файла

Если существует файл `app/static/in_and_out_settings.json`:
1. **Перед циклом:** Файл загружается в словарь `file_data`
2. **Во время цикла:** Для каждого сервиса проверяется наличие данных в файле
3. **Объединение:** Словари `input` и `output` объединяются с данными из файла

### Логика объединения:

```python
# Для каждого сервиса:
combined_input = input_categorized["internal"].copy()  # Из БД
if "input" in file_data[service_id]:
    combined_input.update(file_data[service_id]["input"])  # Расширяем из файла

combined_output = output_categorized["internal"].copy()  # Из БД
if "output" in file_data[service_id]:
    combined_output.update(file_data[service_id]["output"])  # Расширяем из файла
```

## Формат файла

```json
{
  "service_id": {
    "input": {
      "param_name": "widget_type"
    },
    "output": {
      "param_name": "widget_type"
    }
  }
}
```

### Поддерживаемые типы виджетов:

- `"file"` - файловый вход
- `"file_save"` - файловый выход
- `"theme_select"` - выбор темы/датасета

## Пример использования

### app/static/in_and_out_settings.json:

```json
{
  "123": {
    "input": {
      "custom_input_file": "file",
      "dataset_selector": "theme_select"
    },
    "output": {
      "result_raster": "file_save",
      "output_vector": "file_save"
    }
  },
  "456": {
    "input": {
      "source_data": "file"
    },
    "output": {
      "processed_data": "file_save"
    }
  }
}
```

### Результат:

Для сервиса ID=123:
- **Из БД:** `input: {file1: "file"}`, `output: {out1: "file_save"}`
- **Из файла:** `input: {custom_input_file: "file", dataset_selector: "theme_select"}`, `output: {result_raster: "file_save", output_vector: "file_save"}`
- **Итого:** Объединенные словари с данными из обоих источников

## Когда использовать

### ✅ Полезно для:

1. **Дополнения конфигурации БД:**
   - Если параметры не правильно определяются автоматически
   - Если нужно добавить специальные параметры

2. **Переопределения:**
   - Если автоматическое определение работает неправильно
   - Если нужно временно изменить конфигурацию без изменения БД

3. **Тестирования:**
   - Быстрое тестирование новых конфигураций
   - Экспериментирование с композициями

### ❌ Не рекомендуется для:

1. **Основной конфигурации** - лучше обновить БД
2. **Постоянных изменений** - используйте миграции БД
3. **Больших объемов данных** - может замедлить загрузку

## Приоритет данных

Данные из файла имеют **приоритет** и **дополняют** данные из БД:

```
Данные из БД + Данные из файла = Итоговая конфигурация
     ↓                ↓                     ↓
input: {a: "file"} + input: {b: "file"} = input: {a: "file", b: "file"}
```

Если один и тот же ключ есть в обоих источниках:
```
input: {a: "file"} + input: {a: "theme_select"} = input: {a: "theme_select"}
                                                    ↑ приоритет у файла
```

## Создание файла

### Вручную:

```bash
# Создать пустой файл
echo '{}' > app/static/in_and_out_settings.json

# Отредактировать
nano app/static/in_and_out_settings.json
```

### Из примера:

```bash
cp in_and_out_settings.example.json app/static/in_and_out_settings.json
# Отредактировать под ваши нужды
```

## Проверка

После изменения файла:

```bash
# Перезапустить приложение
docker-compose restart app

# Проверить логи
docker-compose logs -f app | grep "Loading data from"

# Должно появиться:
# Loading data from app/static/in_and_out_settings.json...
# Loaded X services from file
```

## Отладка

Если файл не загружается:

```bash
# Проверить синтаксис JSON
python -m json.tool app/static/in_and_out_settings.json

# Проверить права доступа
ls -la app/static/in_and_out_settings.json

# Проверить что файл в правильной директории
pwd
ls -la app/static/in_and_out_settings.json
```

## Примеры

### Пример 1: Добавление файлового параметра

Сервис ID=100 в БД не имеет файловых параметров, но мы знаем что он использует файлы:

```json
{
  "100": {
    "input": {
      "input_raster": "file"
    },
    "output": {
      "output_raster": "file_save"
    }
  }
}
```

### Пример 2: Исправление типа виджета

Сервис ID=200 неправильно определяет тип параметра:

```json
{
  "200": {
    "input": {
      "theme": "theme_select"
    }
  }
}
```

### Пример 3: Множественные параметры

Сервис ID=300 имеет несколько файловых входов и выходов:

```json
{
  "300": {
    "input": {
      "raster1": "file",
      "raster2": "file",
      "vector_layer": "file"
    },
    "output": {
      "result_raster": "file_save",
      "result_vector": "file_save",
      "statistics": "file_save"
    }
  }
}
```

## Важные замечания

1. **Ключи должны быть строками** - JSON требует строковые ключи
2. **ID сервиса должен существовать в БД** - файл только расширяет, не создает новые сервисы
3. **Файл опциональный** - если его нет, используются только данные из БД
4. **Перезапуск требуется** - после изменения файла перезапустите приложение

---

**Последнее обновление:** 10 октября 2025

