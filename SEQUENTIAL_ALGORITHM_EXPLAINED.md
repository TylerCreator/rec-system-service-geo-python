# 🔮 Алгоритм последовательной рекомендации - Подробное объяснение

## 📋 Содержание

1. [Обзор](#обзор)
2. [Шаг 1: Получение compositionsDAG.json](#шаг-1-получение-compositionsdagjson)
3. [Шаг 2: Преобразование в NetworkX DiGraph](#шаг-2-преобразование-в-networkx-digraph)
4. [Шаг 3: Извлечение путей и создание обучающих данных](#шаг-3-извлечение-путей-и-создание-обучающих-данных)
5. [Шаг 4: Кодирование узлов (LabelEncoder)](#шаг-4-кодирование-узлов-labelencoder)
6. [Шаг 5: Создание PyTorch Geometric данных](#шаг-5-создание-pytorch-geometric-данных)
7. [Шаг 6: Обучение DAGNN модели](#шаг-6-обучение-dagnn-модели)
8. [Шаг 7: Inference (предсказание)](#шаг-7-inference-предсказание)

---

## Обзор

Система последовательных рекомендаций предсказывает следующий сервис/таблицу в workflow на основе:
- Исторических композиций из БД
- DAG структуры (направленный ациклический граф)
- Graph Neural Network (DAGNN)

**Время обучения:** 2-5 минут
**Время inference:** 10-50ms

### ⚠️ Важно! Направления связей в DAG

В системе композиций существуют **только два типа связей**:

```
✅ Table → Service
   Таблица передаётся как входной параметр сервису
   Пример: table_1001 → service_123 (параметр "theme")

✅ Service → Service
   Результат одного сервиса передаётся другому сервису
   Пример: service_123 → service_456 (параметр "file")

❌ Service → Table НЕ СУЩЕСТВУЕТ!
   Таблицы НЕ могут быть целью связи
   Таблицы - только источники данных (входные параметры)
```

**Типичный workflow:**
```
table_A → service_1 → service_2
table_B → service_2 → service_3

где:
- table_A, table_B - входные данные
- service_1, service_2, service_3 - обрабатывающие сервисы
- service_1 → service_2 - передача промежуточного результата
```

---

## Шаг 1: Получение compositionsDAG.json

### 1.1 Вызов функции recover_new()

```
┌─────────────────────────────────────────────────┐
│  await recover_new(db)                          │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  1. Загружает все Calls из БД                   │
│     SELECT * FROM Calls ORDER BY id ASC         │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  2. Анализирует входы/выходы каждого вызова     │
│     - call.input  (параметры)                   │
│     - call.result (результаты)                  │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  3. Находит связи через файлы и датасеты        │
│     - WIDGET_FILE: файловые связи               │
│     - WIDGET_THEME_SELECT: датасеты             │
│     - WIDGET_EDIT: промежуточные результаты     │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  4. Строит композиции (workflows)               │
│     Composition = {                             │
│       "nodes": [service1, table1, service2],    │
│       "links": [                                │
│         {source: 0, target: 1},                 │
│         {source: 1, target: 2}                  │
│       ]                                         │
│     }                                           │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  5. Сохраняет в app/static/compositionsDAG.json │
│     [                                           │
│       {nodes: [...], links: [...]},             │
│       {nodes: [...], links: [...]},             │
│       ...                                       │
│     ]                                           │
└─────────────────────────────────────────────────┘
```

### 1.2 Пример compositionsDAG.json

```json
[
  {
    "nodes": [
      {"id": "1002132", "mid": null},      // table_1002132
      {"id": 3985, "mid": 308},            // service_308
      {"id": 3986, "mid": 309}             // service_309
    ],
    "links": [
      {"source": "1002132", "target": 3985, "fields": "1002132:theme"},  // table → service (input)
      {"source": 3985, "target": 3986, "fields": ["file:source"]}        // service → service (result)
    ]
  },
  ...
]
```

**Ключевые поля:**
- `mid` != null - это сервис → конвертируется в `service_{mid}`
- `mid` == null - это таблица → конвертируется в `table_{id}`

**Важно! Направления связей:**
- ✅ **Table → Service** - таблица передается как входной параметр сервису
- ✅ **Service → Service** - результат одного сервиса передается другому
- ❌ **Service → Table** - ТАКОГО НЕ БЫВАЕТ! (таблицы не могут быть целью, только источником)

---

## Шаг 2: Преобразование в NetworkX DiGraph

### 2.1 Зачем нужен NetworkX DiGraph?

```
compositionsDAG.json                NetworkX DiGraph
(список композиций)        →        (единый граф со всеми связями)

[                                    ┌──────────────────────┐
  {nodes: [...], links: [...]},     │   service_123        │
  {nodes: [...], links: [...]},  →  │        ↓             │
  ...                                │   table_1002120      │
]                                    │        ↓             │
                                     │   service_456        │
Проблемы:                            │        ↓             │
- Дублирование узлов                 │   table_1001211      │
- Изолированные графы                └──────────────────────┘
- Нет единой структуры               
                                     Преимущества:
                                     - Единый граф
                                     - Нет дублирования
                                     - Алгоритмы поиска путей
                                     - Анализ связности
```

### 2.2 Процесс преобразования

```python
def _load_dag_from_json(self, json_path: Path) -> nx.DiGraph:
    # Шаг 1: Создать пустой направленный граф
    dag = nx.DiGraph()
    
    # Шаг 2: Для каждой композиции
    for composition in data:
        # Шаг 2.1: Маппинг локальных ID → глобальные имена
        id_to_mid = {}
        for node in composition["nodes"]:
            if "mid" in node:
                # Это сервис
                id_to_mid[node["id"]] = f"service_{node['mid']}"
            else:
                # Это таблица
                id_to_mid[node["id"]] = f"table_{node['id']}"
        
        # Шаг 2.2: Добавить рёбра
        for link in composition["links"]:
            src = id_to_mid[link["source"]]  # Конвертируем локальный ID
            tgt = id_to_mid[link["target"]]  # в глобальное имя
            
            # Добавить узлы с типом
            dag.add_node(src, type='service' if 'service' in src else 'table')
            dag.add_node(tgt, type='service' if 'service' in tgt else 'table')
            
            # Добавить ребро
            dag.add_edge(src, tgt)
    
    return dag
```

### 2.3 Визуализация преобразования

```
Композиция 1:                         Композиция 2:
node_1 (id=1001) → node_2 (mid=123)   node_a (id=1002) → node_b (mid=123)
                                      
После конвертации ID:                 После конвертации ID:
table_1001 → service_123              table_1002 → service_123

                    ↓ ОБЪЕДИНЕНИЕ ↓

           Единый NetworkX DiGraph:
           
         table_1001    table_1002
                  │        │
                  ▼        ▼
           ┌─────────────────────┐
           │   service_123       │
           └─────────────────────┘
         
NetworkX автоматически:
- Объединяет дублирующиеся узлы (service_123 встречается дважды)
- Сохраняет все уникальные рёбра
- Создаёт единую структуру графа
```

### 2.4 Результат

```python
dag.nodes()  # ['service_123', 'table_1001', 'table_1002', 'service_456', ...]
dag.edges()  # [('table_1001', 'service_123'), ('table_1002', 'service_123'), ('service_123', 'service_456'), ...]

# Направления:
# table → service  (таблица как вход)
# service → service (результат передаётся дальше)

# Можно анализировать:
dag.in_degree('table_1001')   # Сколько входящих связей
dag.out_degree('service_123')  # Сколько исходящих связей
dag.successors('table_1001')  # Следующие узлы
nx.has_path(dag, 'table_1001', 'table_1002')  # Есть ли путь
```

---

## Шаг 3: Извлечение путей и создание обучающих данных

### 3.1 Извлечение путей из DAG

```
Дано: DAG граф
┌─────────────────────────────────┐
│  A → B → C → D                  │
│  A → E → F                      │
│  B → G                          │
└─────────────────────────────────┘

Цель: Извлечь все возможные пути

Алгоритм:
for each node in dag:
    if node.out_degree > 0:
        DFS от этого узла
        
Результат - пути:
[
    [A, B, C, D],    # Путь 1
    [A, B, G],       # Путь 2
    [A, E, F],       # Путь 3
    [B, C, D],       # Путь 4
    [B, G],          # Путь 5
    [E, F]           # Путь 6
]
```

### 3.2 Создание обучающих примеров

```
Из путей создаём пары (контекст → следующий узел)

Путь: [table_1001, service_123, table_1002, service_456]

Обучающие примеры:
┌─────────────────────────────────────────────────┐
│ Контекст              →  Следующий узел         │
├─────────────────────────────────────────────────┤
│ (table_1001,)         →  service_123            │
│ (table_1001,          →  table_1002             │
│  service_123)                                   │
│ (table_1001,          →  service_456            │
│  service_123,                                   │
│  table_1002)                                    │
└─────────────────────────────────────────────────┘

Фильтр: Только если следующий узел = service_*
(мы предсказываем только сервисы, таблицы являются входными данными)

Итого:
X_raw = [
    (table_1001,),               # Контекст: начали с таблицы
    (table_1001, service_123),   # Контекст: таблица + сервис
]
y_raw = [
    service_123,  # После table_1001 используют service_123
    service_456   # После (table_1001, service_123) следует service_456
]

**Правильные направления в реальном workflow:**
- table_1001 → service_123 (таблица как вход для сервиса)
- service_123 → service_456 (результат сервиса передаётся дальше)
- table_1002 → service_456 (другая таблица тоже передаётся service_456)

**Важно:** В y_raw мы предсказываем только сервисы, потому что:
  - Сервисы - это то, что пользователь выбирает/запускает
  - Таблицы - это входные параметры, они не могут быть "следующим шагом"
  - В реальности: "После service_123, какой сервис использовать дальше?"
```

### 3.3 Код

```python
def _create_training_data(self, paths: List[List[str]]) -> Tuple[List, List]:
    X_raw = []  # Контексты
    y_raw = []  # Целевые узлы
    
    for path in paths:
        # Для каждой позиции в пути (кроме первой и последней)
        for i in range(1, len(path) - 1):
            context = tuple(path[:i])      # Всё до текущей позиции
            next_step = path[i]            # Текущая позиция
            
            # Только если следующий узел - сервис
            if next_step.startswith("service"):
                X_raw.append(context)
                y_raw.append(next_step)
    
    return X_raw, y_raw
```

**Результат:**
```
X_raw: [(service_123,), (service_123, table_1001), ...]
y_raw: [service_456, table_1002, ...]
```

---

## Шаг 4: Кодирование узлов (LabelEncoder)

### 4.1 Зачем нужен LabelEncoder?

```
Проблема: Neural Networks работают с числами, не строками

Строковые имена узлов:              Числовые индексы:
┌────────────────────┐              ┌──────────────┐
│ service_123        │   →  fit()  │      0       │
│ table_1001         │   →         │      1       │
│ service_456        │   →         │      2       │
│ table_1002         │   →         │      3       │
└────────────────────┘              └──────────────┘

LabelEncoder создаёт биективное отображение:
  string ←→ integer
```

### 4.2 Процесс fit_transform

```python
# Исходные данные
node_list = ['service_123', 'table_1001', 'service_456', 'table_1002']

# Шаг 1: Создать encoder
node_encoder = LabelEncoder()

# Шаг 2: fit_transform
node_ids = node_encoder.fit_transform(node_list)

# Что происходит внутри:
# fit():
#   - Находит все уникальные значения: {'service_123', 'table_1001', ...}
#   - Сортирует их (алфавитно)
#   - Присваивает индексы: 0, 1, 2, 3, ...
#   - Сохраняет маппинг внутри

# transform():
#   - Для каждого значения ищет его индекс
#   - Возвращает массив индексов

# Результат:
node_ids = [0, 1, 2, 3]  # numpy array
```

### 4.3 Создание node_map

```python
# Создаём словарь для быстрого поиска
node_map = {node: idx for node, idx in zip(node_list, node_ids)}

# Результат:
node_map = {
    'service_123': 0,
    'table_1001': 1,
    'service_456': 2,
    'table_1002': 3
}

# Использование:
idx = node_map['service_123']  # → 0
idx = node_map['table_1001']   # → 1
```

### 4.4 Визуальная схема

```
┌──────────────────────────────────────────────────────────┐
│                 LabelEncoder Process                      │
└──────────────────────────────────────────────────────────┘

Input (строки):
  ['service_123', 'table_1001', 'service_456', 'table_1002']
           │
           ▼
  ┌─────────────────┐
  │  LabelEncoder   │
  │                 │
  │  fit():         │
  │   1. Unique     │
  │   2. Sort       │
  │   3. Assign IDs │
  └─────────────────┘
           │
           ▼
Output (числа):
  [0, 1, 2, 3]
  
Mapping (сохранён в encoder):
  service_123 ↔ 0
  table_1001  ↔ 1
  service_456 ↔ 2
  table_1002  ↔ 3

Обратное преобразование (inverse_transform):
  [0, 1, 2, 3] → ['service_123', 'table_1001', ...]
```

---

## Шаг 5: Создание PyTorch Geometric данных

### 5.1 Что такое PyTorch Geometric Data?

```
PyTorch Geometric - библиотека для Graph Neural Networks

Data объект содержит:
┌─────────────────────────────────────┐
│  data.x          # Node features    │
│  data.edge_index # Graph structure  │
│  data.y          # Labels (опц.)    │
└─────────────────────────────────────┘
```

### 5.2 Создание node features (data.x)

```python
# Для каждого узла создаём вектор признаков
features = []
for node in node_list:
    if dag.nodes[node]['type'] == 'service':
        features.append([1, 0])  # One-hot: [is_service, is_table]
    else:
        features.append([0, 1])  # One-hot: [is_service, is_table]

# Преобразуем в тензор
x = torch.tensor(features, dtype=torch.float)
```

**Визуализация:**
```
Узел             Type      Feature Vector
─────────────────────────────────────────
service_123   → service → [1, 0]
table_1001    → table   → [0, 1]
service_456   → service → [1, 0]
table_1002    → table   → [0, 1]

Итоговый тензор x:
tensor([[1, 0],    ← service_123 (индекс 0)
        [0, 1],    ← table_1001  (индекс 1)
        [1, 0],    ← service_456 (индекс 2)
        [0, 1]])   ← table_1002  (индекс 3)

Shape: (4, 2)  # 4 узла, 2 признака каждый
```

### 5.3 Создание edge_index (структура графа)

```python
# Для каждого ребра в DAG
edge_list = []
for u, v in dag.edges():
    src_idx = node_map[u]  # Индекс источника
    tgt_idx = node_map[v]  # Индекс цели
    edge_list.append([src_idx, tgt_idx])

# Преобразуем в тензор и транспонируем
edge_index = torch.tensor(edge_list, dtype=torch.long).t()
```

**Визуализация:**
```
DAG edges (строковые) - ПРАВИЛЬНЫЕ направления:
  table_1001  → service_123  ✅ (таблица как вход для сервиса)
  service_123 → service_456  ✅ (результат сервиса передаётся другому)
  table_1002  → service_456  ✅ (вторая таблица как вход)

Преобразование через node_map:
  service_123: 0
  table_1001:  1
  service_456: 2
  table_1002:  3

Рёбра с индексами:
  1 → 0    (table_1001 → service_123)
  0 → 2    (service_123 → service_456)
  3 → 2    (table_1002 → service_456)

edge_list:
  [[1, 0],
   [0, 2],
   [3, 2]]

edge_index (транспонированный):
  tensor([[1, 0, 3],    ← Source nodes
          [0, 2, 2]])   ← Target nodes

Shape: (2, 3)  # 2 ряда (src, tgt), 3 ребра

Почему транспонирован?
PyTorch Geometric требует формат [2, num_edges]:
  - Ряд 0: индексы источников
  - Ряд 1: индексы целей
```

### 5.4 Создание контекстов и целей для обучения

```python
# Преобразуем X_raw и y_raw в индексы

X_raw = [
    (service_123,),
    (service_123, table_1001),
]

y_raw = [
    service_456,
    table_1002
]

# Используем node_map для конвертации
contexts = torch.tensor([
    node_map[context[-1]]  # Берём ПОСЛЕДНИЙ элемент контекста
    for context in X_raw
], dtype=torch.long)

targets = torch.tensor([
    node_map[y]
    for y in y_raw
], dtype=torch.long)
```

**Визуализация:**
```
X_raw (контексты):                contexts (индексы):
┌──────────────────────┐          ┌──────┐
│ (table_1001,)        │  →       │  1   │  (последний = table_1001)
│ (table_1001,         │  →       │  0   │  (последний = service_123)
│  service_123)        │          │      │
└──────────────────────┘          └──────┘

y_raw (цели):                     targets (индексы):
┌──────────────────────┐          ┌──────┐
│ service_123          │  →       │  0   │
│ table_1002           │  →       │  3   │
└──────────────────────┘          └──────┘

contexts: tensor([1, 0])  # Индексы последних узлов в контекстах
targets:  tensor([0, 3])  # Индексы целевых узлов
```

### 5.5 Итоговый Data объект

```python
from torch_geometric.data import Data

data_pyg = Data(
    x=x,              # Node features: (num_nodes, num_features)
    edge_index=edge_index  # Graph structure: (2, num_edges)
)

# Пример:
data_pyg.x.shape         # torch.Size([4, 2])  - 4 узла, 2 признака
data_pyg.edge_index.shape # torch.Size([2, 3])  - 3 ребра
```

---

## Шаг 6: Обучение DAGNN модели

### 6.1 Архитектура модели

```
┌──────────────────────────────────────────────────────┐
│              DAGNNRecommender Model                   │
└──────────────────────────────────────────────────────┘

Input: x (num_nodes, 2)  +  edge_index (2, num_edges)
  │
  ▼
┌─────────────────────────────────────────┐
│ Layer 1: Linear(2 → 64)                 │
│          + BatchNorm                    │
│          + ReLU                         │
│          + Dropout(0.4)                 │
└───────────────┬─────────────────────────┘
                │ (num_nodes, 64)
                ▼
┌─────────────────────────────────────────┐
│ Layer 2: Linear(64 → 64)                │
│          + BatchNorm                    │
│          + ReLU                         │
│          + Residual Connection          │
│          + Dropout(0.4)                 │
└───────────────┬─────────────────────────┘
                │ (num_nodes, 64)
                ▼
┌─────────────────────────────────────────┐
│ Layer 3: DAGNN(64, K=10)                │
│          Graph Propagation              │
│          + Attention Weights            │
└───────────────┬─────────────────────────┘
                │ (num_nodes, 64)
                ▼
┌─────────────────────────────────────────┐
│ Layer 4: Linear(64 → 32)                │
│          + BatchNorm                    │
│          + ReLU                         │
│          + Dropout(0.4)                 │
└───────────────┬─────────────────────────┘
                │ (num_nodes, 32)
                ▼
┌─────────────────────────────────────────┐
│ Layer 5: Linear(32 → num_nodes)         │
│          Output layer                   │
└───────────────┬─────────────────────────┘
                │ (num_nodes, num_nodes)
                ▼
Output: Predictions for each node
```

### 6.2 Детальное прохождение через слои

**Пример:** 4 узла, 2 признака

#### Вход:
```
x: tensor([[1, 0],    # service_123
           [0, 1],    # table_1001
           [1, 0],    # service_456
           [0, 1]])   # table_1002
Shape: (4, 2)
```

#### Layer 1: Linear(2 → 64)
```python
self.lin1 = nn.Linear(2, 64)

# Математика:
out = x @ W^T + b
# где W: (64, 2), b: (64,)

# Результат:
x = self.lin1(x)  # (4, 2) → (4, 64)
x = self.bn1(x)   # BatchNorm нормализует по батчу
x = F.relu(x)     # ReLU: max(0, x)
x = F.dropout(x, 0.4)  # Случайно зануляет 40% нейронов

После Layer 1:
tensor([[0.23, 0.45, 0.0, 0.89, ...],   # service_123 (64 значения)
        [0.12, 0.0, 0.56, 0.34, ...],   # table_1001
        [0.67, 0.23, 0.12, 0.0, ...],   # service_456
        [0.0, 0.78, 0.23, 0.45, ...]])  # table_1002
Shape: (4, 64)
```

#### Layer 2: Linear(64 → 64) + Residual
```python
identity = x  # Сохраняем для residual connection

x = self.lin2(x)      # (4, 64) → (4, 64)
x = self.bn2(x)       # BatchNorm
x = F.relu(x)         # ReLU
x = x + identity      # Residual: добавляем исходный x
x = F.dropout(x, 0.4)

Residual Connection (почему важно):
┌────────┐
│   x    │ ──────────┐
└───┬────┘           │
    │                │
    ▼                │
┌────────┐           │
│ Linear │           │
│  + BN  │           │
│ + ReLU │           │
└───┬────┘           │
    │                │
    └────── + ◄──────┘
         (element-wise add)
    
Помогает избежать vanishing gradient
```

#### Layer 3: DAGNN (Graph Propagation)

```python
self.dagnn = DAGNN(64, K=10, dropout=0.4)

# DAGNN - это APPNP (Approximate Personalized Propagation of Neural Predictions)
```

**Что делает DAGNN:**

```
Итерация 0: x₀ = x (начальные features)

Итерация 1: x₁ = propagate(x₀, edge_index)
  Для каждого узла: x₁[i] = Σ(x₀[neighbors] * weights)
  
Итерация 2: x₂ = propagate(x₁, edge_index)
  ...
  
Итерация K: x_K = propagate(x_{K-1}, edge_index)

Финальный выход:
  out = attention_weights[0] * x₀ + 
        attention_weights[1] * x₁ + 
        ... + 
        attention_weights[K] * x_K
```

**Визуально (K=2):**

```
Граф:
  0 → 1 → 2 → 3

Итерация 0 (исходные features):
  node_0: [0.5, 0.3, ...]
  node_1: [0.2, 0.7, ...]
  node_2: [0.8, 0.1, ...]
  node_3: [0.4, 0.6, ...]

Итерация 1 (первая propagation):
  node_0: [0.5, 0.3, ...] (нет входящих)
  node_1: avg([0.5, 0.3, ...]) = [0.5, 0.3, ...]  от node_0
  node_2: avg([0.2, 0.7, ...]) = [0.2, 0.7, ...]  от node_1
  node_3: avg([0.8, 0.1, ...]) = [0.8, 0.1, ...]  от node_2

Итерация 2 (вторая propagation):
  node_0: [0.5, 0.3, ...]
  node_1: avg([0.5, 0.3, ...])
  node_2: avg([0.5, 0.3, ...])  ← информация от node_0 дошла!
  node_3: avg([0.2, 0.7, ...])  ← информация от node_1 дошла!

Финальный выход (взвешенная сумма):
  out = 0.3 * x₀ + 0.5 * x₁ + 0.2 * x₂
  (attention_weights обучаются!)
```

**Зачем это нужно:**
- Узлы "узнают" о соседях
- Информация распространяется по графу
- Дальние узлы влияют друг на друга

#### Layer 4: Linear(64 → 32)
```python
x = self.lin3(x)      # (4, 64) → (4, 32)
x = self.bn3(x)       # BatchNorm
x = F.relu(x)         # ReLU
x = F.dropout(x, 0.4)

Уменьшаем размерность для финального предсказания
```

#### Layer 5: Output Linear(32 → num_nodes)
```python
x = self.lin_out(x)   # (4, 32) → (4, 4)

# Для каждого узла получаем вероятности переходов во все узлы

Output (logits):
tensor([[-0.5,  2.1, -1.3,  0.8],   # service_123: вероятности → [table_1001, service_456, ...]
        [ 1.2, -0.3,  3.5, -0.8],   # table_1001: ...
        [-0.9,  0.7, -1.1,  2.9],   # service_456: ...
        [ 0.3, -1.5,  1.8, -0.4]])  # table_1002: ...

Shape: (4, 4)  # Каждый узел → вероятности для каждого узла
```

### 5.6 Полная схема прямого прохода

```
Input:
  x: (4, 2)
  edge_index: (2, 3)
  
     ↓ Linear(2→64) + BN + ReLU + Dropout
     
  (4, 64)
  
     ↓ Linear(64→64) + BN + ReLU + Residual + Dropout
     
  (4, 64)
  
     ↓ DAGNN(K=10) - Graph Propagation
     
  (4, 64)
  
     ↓ Linear(64→32) + BN + ReLU + Dropout
     
  (4, 32)
  
     ↓ Linear(32→4)
     
  (4, 4) ← Output logits

     ↓ Softmax (при inference)
     
  (4, 4) ← Probabilities
```

---

## Шаг 6.3: Процесс обучения

### 6.3.1 Loss Calculation

```python
# Forward pass
out = model(data_pyg.x, data_pyg.edge_index, training=True)
# out shape: (num_nodes, num_nodes)

# Выбираем только нужные узлы (из контекстов)
out_selected = out[contexts]
# Если contexts = [0, 1], то out_selected = [out[0], out[1]]

# Вычисляем Cross Entropy Loss
loss = F.cross_entropy(out_selected, targets)
```

**Визуальная схема:**

```
out (все узлы):
  [[logit_0_0, logit_0_1, logit_0_2, logit_0_3],   ← node_0
   [logit_1_0, logit_1_1, logit_1_2, logit_1_3],   ← node_1
   [logit_2_0, logit_2_1, logit_2_2, logit_2_3],   ← node_2
   [logit_3_0, logit_3_1, logit_3_2, logit_3_3]]   ← node_3

contexts = [1, 0]  # Индексы узлов из которых предсказываем

out[contexts]:
  [[logit_1_0, logit_1_1, logit_1_2, logit_1_3],   ← node_1 (table_1001)
   [logit_0_0, logit_0_1, logit_0_2, logit_0_3]]   ← node_0 (service_123)

targets = [0, 2]  # Правильные ответы: service_123 и service_456

Cross Entropy:
  loss = -log(softmax(out[1])[0]) +    # table_1001 → service_123
         -log(softmax(out[0])[2])      # service_123 → service_456
  
  Цель: Минимизировать loss
  → Модель должна давать высокие значения для правильных классов
```

### 6.3.2 Backward Pass & Optimization

```python
# 1. Обнулить градиенты
optimizer.zero_grad()

# 2. Forward pass
out = model(data_pyg.x, data_pyg.edge_index, training=True)
loss = F.cross_entropy(out[contexts], targets)

# 3. Backward pass (вычисление градиентов)
loss.backward()
# Вычисляет ∂loss/∂W для всех весов W

# 4. Gradient clipping (предотвращает exploding gradients)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 5. Update weights
optimizer.step()
# W_new = W_old - learning_rate * gradient
```

**Пример обучения на реальных данных:**

```
Обучающая выборка:
  table_1001 → service_123  (таблица передаётся сервису)
  service_123 → service_456 (результат сервиса идёт дальше)
  table_1002 → service_456  (вторая таблица для сервиса)

contexts = [1, 0, 3]  # Индексы: table_1001, service_123, table_1002
targets  = [0, 2, 2]  # Индексы: service_123, service_456, service_456

Модель учится:
  "После table_1001 обычно используют service_123"
  "После service_123 обычно используют service_456"
  "После table_1002 обычно используют service_456"
```

**Визуализация обучения:**

```
Epoch 1:
┌────────────────────────────────────────┐
│ Forward  → Loss: 2.35                  │
│ Backward → Gradients computed          │
│ Update   → Weights updated             │
└────────────────────────────────────────┘
         │
         ▼ Weights improved
         
Epoch 2:
┌────────────────────────────────────────┐
│ Forward  → Loss: 1.89  ✓ (lower!)     │
│ Backward → Gradients computed          │
│ Update   → Weights updated             │
└────────────────────────────────────────┘
         │
         ▼ Weights improved more
         
...

Epoch 200:
┌────────────────────────────────────────┐
│ Forward  → Loss: 0.15  ✓✓✓ (converged)│
│ EARLY STOPPING (no improvement)        │
└────────────────────────────────────────┘
```

### 6.3.3 Детали обучения каждого слоя

**Layer 1 (Linear + BN):**
```
Обучаемые параметры:
  W1: (64, 2)    - веса linear слоя
  b1: (64,)      - bias linear слоя
  γ1: (64,)      - scale BatchNorm
  β1: (64,)      - shift BatchNorm

Итого: 64*2 + 64 + 64 + 64 = 320 параметров

Что учится:
  - Какие комбинации признаков [is_service, is_table] важны
  - Как преобразовать 2D в 64D пространство
```

**Layer 2 (Linear + Residual):**
```
Обучаемые параметры:
  W2: (64, 64)   - веса
  b2: (64,)      - bias
  γ2: (64,)      - BatchNorm scale
  β2: (64,)      - BatchNorm shift

Итого: 64*64 + 64 + 64 + 64 = 4288 параметров

Что учится:
  - Более сложные паттерны в 64D пространстве
  - Residual помогает сохранить информацию из Layer 1
```

**Layer 3 (DAGNN - Graph Propagation):**
```
Обучаемые параметры:
  attention_weights: (K+1,)  где K=10
  
Итого: 11 параметров

Что учится:
  - Сколько внимания уделять каждой итерации propagation
  - att[0]: вес исходных features
  - att[1]: вес после 1 hop
  - att[2]: вес после 2 hops
  - ...
  - att[10]: вес после 10 hops
  
Softmax(att) гарантирует сумма = 1
```

**DAGNN Propagation детально:**

```
K = 10 итераций

Итерация 0: x₀ = current_features

Итерация 1:
  for each node i:
    x₁[i] = Σ(x₀[neighbor_j] * edge_weight[j→i]) / degree[i]
    
  Пример для node_1 (table_1001):
    neighbors = [node_0]  # service_123
    x₁[1] = x₀[0] * 1.0 / 1 = x₀[0]
    
  Для node_2 (service_456):
    neighbors = [node_1]
    x₁[2] = x₀[1]

Итерация 2:
  node_2 теперь получает информацию от node_0 (через node_1)
  x₂[2] = aggregate(x₁[1]) = aggregate(x₀[0])
  
... продолжается K раз

Финал:
  out = att[0]*x₀ + att[1]*x₁ + ... + att[10]*x₁₀
  
Каждый узел "видит" информацию на расстоянии до K hops!
```

**Layer 4 (Dimension Reduction):**
```
Обучаемые параметры:
  W4: (32, 64)
  b4: (32,)
  γ4, β4: BatchNorm

Итого: 32*64 + 32 + 32 + 32 = 2144 параметров

Что учится:
  - Сжатие 64D → 32D (убирает шум)
  - Выделяет наиболее важные признаки
```

**Layer 5 (Output):**
```
Обучаемые параметры:
  W_out: (num_nodes, 32)  # В нашем примере (4, 32)
  b_out: (num_nodes,)     # (4,)

Итого: 4*32 + 4 = 132 параметров

Что учится:
  - Для каждого возможного целевого узла
  - Какие паттерны в 32D пространстве указывают на этот узел
```

### 6.3.4 Итого параметров модели

```
Layer 1:      320 параметров
Layer 2:    4,288 параметров
DAGNN:         11 параметров
Layer 4:    2,144 параметров
Layer 5:      132 параметров
─────────────────────────────
ИТОГО:     6,895 параметров

Все они обучаются одновременно через backpropagation!
```

### 6.3.5 Процесс обучения (200 эпох)

```
Epoch 1:
  Loss: 2.347
  Gradient: Large (модель далека от оптимума)
  Learning rate: 0.001
  Weight update: Значительный
  
Epoch 10:
  Loss: 1.523  ✓
  Gradient: Medium
  Learning rate: 0.001 (не изменился)
  
Epoch 50:
  Loss: 0.823  ✓✓
  Gradient: Small
  Learning rate: 0.0005  ← ReduceLROnPlateau снизил
  
Epoch 100:
  Loss: 0.234  ✓✓✓
  Gradient: Very small
  Learning rate: 0.00025
  
Epoch 150:
  Loss: 0.152
  No improvement for 50 epochs
  → EARLY STOPPING

Модель обучена!
```

---

## Шаг 7: Inference (предсказание)

### 7.1 Процесс предсказания

```
Дано: sequence = [table_1001, service_123]
Цель: Предсказать следующий узел

Шаг 1: Взять последний узел
  last_node = 'service_123'
  last_idx = node_map['service_123'] = 0

Шаг 2: Forward pass через модель
  model.eval()  # Отключить dropout
  with torch.no_grad():  # Не вычислять градиенты
      out = model(data_pyg.x, data_pyg.edge_index, training=False)
  
  out shape: (4, 4)
  out = [
    [logit_0_to_0, logit_0_to_1, logit_0_to_2, logit_0_to_3],
    [logit_1_to_0, logit_1_to_1, logit_1_to_2, logit_1_to_3],  ← нам нужна эта строка!
    [logit_2_to_0, logit_2_to_1, logit_2_to_2, logit_2_to_3],
    [logit_3_to_0, logit_3_to_1, logit_3_to_2, logit_3_to_3]
  ]

Шаг 3: Взять предсказания для последнего узла
  node_scores = out[1]  # Строка для table_1001
  # tensor([-0.5, 0.3, 2.8, 0.9])

Шаг 4: Применить Softmax
  probs = F.softmax(node_scores, dim=0)
  # tensor([0.05, 0.11, 0.72, 0.12])
  #         ↑     ↑     ↑     ↑
  #      node_0 node_1 node_2 node_3

Шаг 5: Сортировать по вероятности
  top_indices = np.argsort(probs)[::-1]
  # [2, 3, 1, 0]  ← node_2 (service_456) наиболее вероятен!

Шаг 6: Преобразовать обратно в имена
  predictions = [
    reverse_node_map[2],  # service_456
    reverse_node_map[3],  # table_1002
    ...
  ]
```

### 7.2 Детальный пример

```
DAG (ПРАВИЛЬНЫЕ направления):
  table_1001  → service_123
  table_1002  → service_456
  service_123 → service_456
  service_456 → service_789

Последовательность: [table_1001, service_123]
Последний узел: service_123 (индекс 0)

Forward pass:
  out[0] = [
     0.3,   # service_123 → service_123 (unlikely, петля)
    -0.5,   # service_123 → table_1001  (IMPOSSIBLE! table не может быть целью)
     2.8,   # service_123 → service_456 (LIKELY! ✓)
     0.9,   # service_123 → service_789 (possible, но менее вероятно)
     ...
  ]

Softmax:
  probs = [0.11, 0.05, 0.72, 0.12, ...]
                         ↑
                   service_456 (72% вероятность)

Top-5 predictions (фильтруем только сервисы):
  1. service_456  (0.72) ← Модель правильно предсказала!
  2. service_789  (0.12)
  3. service_123  (0.11) (повторное использование)
  4. ...
  
Note: table_1001 исключается из предсказаний 
      (таблицы не могут быть целью, только источником!)
```

---

## Особенности Table-based режима

### Отличия от Service-based

```
Service-based:                    Table-based:
┌──────────────────────┐         ┌──────────────────────┐
│ Sequence:            │         │ Sequence:            │
│ [s_123, s_456, s_789]│         │ [t_1001, t_1002]     │
└──────────────────────┘         └──────────────────────┘
         │                                │
         ▼                                ▼
┌──────────────────────┐         ┌──────────────────────┐
│ Предсказать:         │         │ Найти достижимые:    │
│ service_999          │         │ Все таблицы из       │
│                      │         │ t_1002 через любые   │
│ Учитывает:           │         │ промежуточные        │
│ - Только сервисы     │         │ сервисы              │
└──────────────────────┘         └──────────────────────┘

Пример (ПРАВИЛЬНАЯ схема):
t_1001 → s_123 → s_456
t_1002 → s_456 → s_789

s_789 использует/создаёт → t_1003

Service sequence: [s_123, s_456]
  → Predict: s_789

Table sequence: [t_1001, t_1002]
  → Анализ:
    1. t_1002 используется в s_456
    2. После s_456 идёт s_789
    3. s_789 связан с t_1003
  → Найти: t_1003
  → Distance: 1 (от t_1002 до t_1003 - один table hop)
```

### Алгоритм поиска достижимых таблиц

```python
def _find_reachable_tables(start_table):
    """BFS по графу, собирая только таблицы"""
    
    reachable = set()
    visited = set()
    queue = [start_table]
    
    while queue:
        current = queue.pop(0)
        visited.add(current)
        
        # Смотрим всех наследников
        for successor in dag.successors(current):
            # Если таблица - добавляем в результат
            if successor.startswith("table_"):
                reachable.add(successor)
            
            # Продолжаем поиск через сервисы тоже
            if successor not in visited:
                queue.append(successor)
    
    return reachable
```

**Визуально:**

```
Start: table_1001

Реальный DAG:
  table_1001 → service_123
  service_123 → service_456
  table_1002 → service_456
  table_1003 → service_789
  service_456 → service_789

Итерация 1:
  current = table_1001
  successors = [service_123]  # Таблица используется в service_123
  reachable = []              # service_123 не таблица, не добавляем
  queue = [service_123]

Итерация 2:
  current = service_123
  successors = [service_456]  # Результат идёт в service_456
  reachable = []
  queue = [service_456]

Итерация 3:
  current = service_456
  successors = [service_789]
  reachable = []
  queue = [service_789]

Итерация 4:
  current = service_789
  successors = []  # Конечный сервис
  
  НО! Проверяем все таблицы в графе:
    table_1002 → service_456 (service_456 уже посещён)
    table_1003 → service_789 (service_789 уже посещён)
  
  reachable = [table_1002, table_1003]  ← Таблицы для посещённых сервисов!

Результат: [table_1002, table_1003]

Объяснение:
  От table_1001 мы прошли через service_123 → service_456 → service_789
  Эти сервисы используют/связаны с table_1002 и table_1003
  Значит table_1002 и table_1003 - возможные следующие таблицы!
```

---

## Полная схема работы системы

```
┌─────────────────────────────────────────────────────────────┐
│                    SEQUENTIAL RECOMMENDATION                 │
│                         FULL PIPELINE                        │
└─────────────────────────────────────────────────────────────┘

1️⃣ СБОР ДАННЫХ
   ┌──────────────┐
   │  PostgreSQL  │
   │    Calls     │
   └──────┬───────┘
          │
          ▼
   recover_new(db)
          │
          ▼
   ┌──────────────────┐
   │compositionsDAG.  │
   │     json         │
   └──────┬───────────┘

2️⃣ ПОСТРОЕНИЕ ГРАФА
          │
          ▼
   _load_dag_from_json()
          │
          ▼
   ┌──────────────────┐
   │  NetworkX DiGraph│
   │  - nodes: 1247   │
   │  - edges: 3421   │
   └──────┬───────────┘

3️⃣ СОЗДАНИЕ ОБУЧАЮЩИХ ДАННЫХ
          │
          ▼
   _extract_paths()
          │
          ▼
   Paths: [[A,B,C], [D,E,F], ...]
          │
          ▼
   _create_training_data()
          │
          ▼
   X_raw: [(A,), (A,B), ...]
   y_raw: [C, D, ...]

4️⃣ КОДИРОВАНИЕ
          │
          ▼
   LabelEncoder.fit_transform()
          │
          ▼
   node_map: {node_name: index}
          │
          ▼
   contexts: tensor([0, 5, 12, ...])
   targets:  tensor([3, 8, 15, ...])

5️⃣ СОЗДАНИЕ PYTORCH GEOMETRIC DATA
          │
          ▼
   Data(
     x = [[1,0], [0,1], ...],        # Node features
     edge_index = [[0,1,...],        # Graph structure
                   [1,2,...]]
   )

6️⃣ ОБУЧЕНИЕ
          │
          ▼
   for epoch in range(200):
     ├─ Forward pass
     ├─ Calculate loss
     ├─ Backward pass
     ├─ Update weights
     └─ Check early stopping

7️⃣ СОХРАНЕНИЕ
          │
          ▼
   Save:
   - dagnn_model.pth      (веса модели)
   - dagnn_metadata.pkl   (node_map, DAG)

8️⃣ INFERENCE
          │
          ▼
   Input: sequence = [123, 456]
          │
          ▼
   last_node_idx = node_map['service_456']
          │
          ▼
   out = model(x, edge_index)
   probs = softmax(out[last_node_idx])
          │
          ▼
   top_predictions = argsort(probs)[:n]
          │
          ▼
   Output: [service_789, service_321, ...]
```

---

## Математические формулы

### Forward Pass

```
Input: x ∈ ℝ^(N×2), edge_index ∈ ℤ^(2×E)
где N = число узлов, E = число рёбер

Layer 1:
  h₁ = ReLU(BN(xW₁ᵀ + b₁))
  h₁ ∈ ℝ^(N×64)

Layer 2 (с residual):
  h₂ = ReLU(BN(h₁W₂ᵀ + b₂)) + h₁
  h₂ ∈ ℝ^(N×64)

DAGNN Propagation:
  h₀ = h₂
  
  for k = 1 to K:
    hₖ = D⁻¹Ahₖ₋₁
    где A - adjacency matrix, D - degree matrix
  
  h_out = Σₖ αₖhₖ
  где α = softmax(att_weights)

Layer 4:
  h₄ = ReLU(BN(h_outW₄ᵀ + b₄))
  h₄ ∈ ℝ^(N×32)

Layer 5 (output):
  ŷ = h₄W_outᵀ + b_out
  ŷ ∈ ℝ^(N×N)

Loss:
  L = -Σᵢ log(softmax(ŷᵢ)[yᵢ])
  где i пробегает по training samples
```

### Backpropagation

```
Градиенты вычисляются от output к input:

∂L/∂W_out ← вычисляется первым
   ↓
∂L/∂W₄
   ↓
∂L/∂att_weights (DAGNN)
   ↓
∂L/∂W₂
   ↓
∂L/∂W₁ ← вычисляется последним

Обновление весов:
  W_new = W_old - lr * ∂L/∂W

где lr = learning_rate (0.001 → 0.0005 → 0.00025)
```

---

## Сравнение: Services vs Tables режимы

### Services Mode

```
Input: [service_123, service_456]

Algorithm:
  1. last_service = service_456
  2. model.predict(service_456)
  3. Filter: only service_* nodes
  4. Return: [service_789, service_321, ...]
  
Output: Следующие сервисы
```

### Tables Mode

```
Input: [table_1001, table_1002]

Algorithm:
  1. last_table = table_1002
  2. reachable = find_reachable_tables(table_1002)
     → [table_1003, table_1004, ...] (через BFS)
  3. For each reachable_table:
       a. model.predict(last_table) → model_score
       b. calculate_distance(last_table, reachable_table) → distance
       c. get_frequency(reachable_table) → frequency
       d. combined_score = 0.5*model + 0.3*distance + 0.2*freq
  4. Sort by combined_score
  5. Return top N
  
Output: Следующие таблицы с учётом:
  - ML prediction
  - Расстояние в DAG
  - Частота использования
```

---

## Оптимизации и трюки

### 1. Batch Normalization
```
Зачем: Стабилизирует обучение
Как работает:
  - Нормализует входы: (x - mean) / std
  - Обучаемые параметры γ, β для scaling/shifting
  - Помогает избежать internal covariate shift
```

### 2. Dropout
```
Зачем: Предотвращает overfitting
Как работает:
  - Training: Случайно зануляет 40% нейронов
  - Inference: Все нейроны активны, выходы * (1-p)
  - Заставляет сеть учить robustные features
```

### 3. Residual Connection
```
Зачем: Помогает градиентам "течь" назад
Как работает:
  out = F(x) + x
  - Если F(x) учится плохо, градиент всё равно проходит через x
  - Помогает обучать глубокие сети
```

### 4. Gradient Clipping
```
Зачем: Предотвращает exploding gradients
Как работает:
  if ||gradient|| > max_norm:
      gradient = gradient * max_norm / ||gradient||
  - Ограничивает максимальную норму градиента
  - max_norm = 1.0 в нашем случае
```

### 5. Early Stopping
```
Зачем: Останавливает обучение когда нет улучшений
Как работает:
  if loss < best_loss:
      best_loss = loss
      patience_counter = 0
  else:
      patience_counter += 1
      if patience_counter >= 50:
          STOP  # 50 эпох без улучшения
```

### 6. Learning Rate Scheduling
```
Зачем: Адаптивно уменьшает learning rate
Как работает:
  ReduceLROnPlateau:
    if loss не улучшается 20 эпох:
        lr = lr * 0.5
    
  0.001 → 0.0005 → 0.00025 → ...
  
  Помогает fine-tune в конце обучения
```

---

## Итоговая статистика

```
Входные данные:
  - Compositions: ~500-1000
  - Nodes: ~1000-2000
  - Edges: ~2000-5000
  - Training samples: ~1000-3000

Модель:
  - Параметры: ~7000
  - Слои: 5
  - Типы: Linear, BatchNorm, DAGNN

Обучение:
  - Эпохи: 50-200
  - Время: 2-5 минут
  - Early stopping: обычно ~100-150 эпох

Inference:
  - Время: 10-50ms
  - Память: ~50MB (модель в RAM)
```

---

## Схема потока данных

```
┌─────────┐
│   DB    │ Calls table
└────┬────┘
     │
     ▼
┌─────────────┐
│recover_new()│ Анализ связей
└─────┬───────┘
      │
      ▼
┌──────────────────┐
│compositionsDAG.  │ JSON file
│     json         │
└─────┬────────────┘
      │
      ▼
┌──────────────────┐
│  nx.DiGraph()    │ Граф структура
└─────┬────────────┘
      │
      ├─────────────────────┬─────────────────────┐
      │                     │                     │
      ▼                     ▼                     ▼
┌──────────┐        ┌──────────┐        ┌──────────┐
│ Extract  │        │ Label    │        │ Features │
│ Paths    │        │ Encoding │        │ Creation │
└────┬─────┘        └────┬─────┘        └────┬─────┘
     │                   │                    │
     ▼                   ▼                    ▼
  X_raw,y_raw      node_map: {}          x: [[1,0],
                   {str→int}                  [0,1],
                                              ...]
     │                   │                    │
     └───────────────────┴────────────────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │ PyG Data    │
                  │ x + edges   │
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │   DAGNN     │
                  │  Training   │
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │Save Model   │
                  │  .pth + .pkl│
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │  Inference  │
                  │  Ready!     │
                  └─────────────┘
```

---

## Заключение

Система последовательных рекомендаций использует современные Graph Neural Networks для анализа workflow паттернов. Ключевые компоненты:

1. **DAG структура** - единое представление всех композиций
2. **LabelEncoder** - преобразование узлов в числа
3. **PyTorch Geometric** - эффективная работа с графами
4. **DAGNN** - propagation информации по графу
5. **Two modes** - Services и Tables для разных задач

**Результат: Точные предсказания следующего шага в workflow! 🎯**

