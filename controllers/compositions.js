const axios = require("axios");
const { connectionManager } = require("../db");
const models = require("../models/models"); // Путь к модели
const fs = require("fs");
const path = require("path");
const { raw } = require("body-parser");

// Константы
const DATASET_ID_OFFSET = 1000000;
const WIDGET_TYPES = {
    FILE: 'file',
    FILE_SAVE: 'file_save',
    THEME_SELECT: 'theme_select'
};
const TASK_STATUS = {
    SUCCEEDED: 'TASK_SUCCEEDED'
};

// Утилиты для работы с параметрами сервисов
const parseServiceParams = (paramsString) => {
    try {
        return JSON.parse(paramsString);
    } catch (error) {
        console.error('Error parsing service params:', error);
        return [];
    }
};

const categorizeParams = (params, inputWidgetTypes = [WIDGET_TYPES.FILE], outputWidgetTypes = [WIDGET_TYPES.FILE_SAVE]) => {
    const external = {};
    const internal = params.reduce((acc, param) => {
        const widgetName = param.widget?.name;
        
        if (inputWidgetTypes.includes(widgetName) || outputWidgetTypes.includes(widgetName)) {
            acc[param.fieldname] = widgetName;
        } else {
            external[param.fieldname] = param.type;
        }
        
        return acc;
    }, {});
    
    return { internal, external };
};

// Создание объекта inAndOut для анализа связей между сервисами
const buildServiceConnectionMap = async () => {
    console.log('Building service connection map...');
    const serviceData = await models.Service.findAll();
    
    const inAndOut = serviceData.reduce((acc, service) => {
        const params = parseServiceParams(service.params);
        const outputParams = parseServiceParams(service.output_params);
        
        const { internal: input, external: externalInput } = categorizeParams(
            params, 
            [WIDGET_TYPES.FILE, WIDGET_TYPES.THEME_SELECT]
        );
        
        const { internal: output, external: externalOutput } = categorizeParams(
            outputParams, 
            [], 
            [WIDGET_TYPES.FILE_SAVE]
        );
        
        acc[service.id] = {
            type: service.type,
            name: service.name,
            input,
            externalInput,
            output,
            externalOutput
        };
        
        return acc;
    }, {});
    
    // Сохраняем результат в файл
    const filePath = __dirname + "/inAndOut.json";
    fs.writeFile(filePath, JSON.stringify(inAndOut), (err) => {
        if (err) {
            console.error('Error writing inAndOut.json:', err);
        } else {
            console.log("Service connection map saved to inAndOut.json");
        }
    });
    
    return inAndOut;
};

/**
 * Создание маппинга GUID к ID для датасетов
 * Некоторые датасеты идентифицируются по GUID, а не по числовому ID
 * Эта функция создает справочник для быстрого преобразования
 * 
 * @returns {Object} Маппинг в формате { "guid_string": numeric_id }
 */
const buildDatasetGuidMap = async () => {
    console.log('Building dataset GUID to ID mapping...');
    const datasets = await models.Dataset.findAll();
    
    return datasets.reduce((acc, dataset) => {
        if (dataset.guid) {
            acc[dataset.guid] = dataset.id;
        }
        return acc;
    }, {});
};

/**
 * Безопасный парсинг JSON с обработкой ошибок
 * Предотвращает падение приложения при некорректном JSON
 * 
 * @param {string} jsonString - Строка JSON для парсинга
 * @param {*} defaultValue - Значение по умолчанию при ошибке
 * @returns {*} Распарсенный объект или значение по умолчанию
 */
const safeJsonParse = (jsonString, defaultValue = {}) => {
    try {
        return JSON.parse(jsonString);
    } catch (error) {
        console.error('Error parsing JSON:', error);
        return defaultValue;
    }
};

/**
 * Нормализация ID датасета для единообразной обработки
 * Преобразует GUID в числовые ID и добавляет offset для избежания коллизий
 * 
 * @param {string|number} datasetId - Исходный ID или GUID датасета
 * @param {Object} guidMap - Маппинг GUID -> ID
 * @returns {number} Нормализованный числовой ID с offset
 */
const normalizeDatasetId = (datasetId, guidMap) => {
    let normalizedId = datasetId;
    
    // Конвертируем GUID в ID если необходимо
    if (guidMap[datasetId]) {
        normalizedId = guidMap[datasetId];
    }
    
    // Конвертируем в число если это строка
    if (typeof normalizedId === 'string') {
        normalizedId = parseInt(normalizedId, 10);
    }
    
    // Добавляем offset для отличия от service ID
    // Это предотвращает конфликты между ID датасетов и сервисов в графе
    return normalizedId + DATASET_ID_OFFSET;
};

async function createCompositions(compositions) {
    try {
        for (const compositionData of compositions) {
            await createCompositionInDatabase(compositionData);
        }
        // res.send("Композиции успешно обновлены");
        console.log("Композиции успешно обновлены");
    } catch (error) {
        console.error("Ошибка при обновлении композиций:", error);
        // res.status(500).send("Ошибка при обновлении композиций");
        return error;
    }
}
async function createCompositionInDatabase(compositionData) {
    try {
        // Создайте запись композиции в базе данных, передав объект compositionData
        const newComposition = await models.Composition.findOrCreate({
            where: { id: compositionData.id },
            defaults: compositionData,
        });
        console.log("Создана новая композиция:", newComposition);
    } catch (error) {
        console.error("Ошибка при создании композиции:", error);
    }
}

async function createUsers(users) {
    try {
        for (const user in users) {
            const [newUser, created] = await models.User.findOrCreate({
                where: { id: user },
                defaults: {
                    id: user,
                },
            });
        }
        // res.send("Композиции успешно обновлены");
        console.log("Пользователь успешно создан");
    } catch (error) {
        console.error("Ошибка при создании пользователя:", error);
        // res.status(500).send("Ошибка при обновлении композиций");
        return error;
    }
}

// Добавление связи между задачами
const addTaskLink = (links, taskId, sourceId, sourceParamName, paramName) => {
    const linkData = {
        source: sourceId,
        target: taskId,
        value: `${sourceParamName}:${paramName}`
    };
    
    if (links[taskId]) {
        links[taskId].push(linkData);
    } else {
        links[taskId] = [linkData];
    }
};

// Создание узла композиции
const createCompositionNode = (task, inAndOut) => {
    const inputs = safeJsonParse(task.input, {});
    const outputs = safeJsonParse(task.result, {});
    const serviceInfo = inAndOut[task.mid];
    
    if (!serviceInfo) return null;
    
    const localInputs = Object.keys(serviceInfo.externalInput).map(inputName => ({
        name: inputName,
        value: inputs[inputName],
        type: serviceInfo.externalInput[inputName]
    }));
    
    const localOutputs = Object.keys(serviceInfo.externalOutput).map(outputName => ({
        name: outputName,
        value: outputs[outputName],
        type: serviceInfo.externalOutput[outputName]
    }));
    
    return {
        mid: task.mid,
        taskId: task.id,
        type: serviceInfo.type,
        service: serviceInfo.name,
        owner: task.owner,
        inputs: localInputs,
        outputs: localOutputs,
        end_time: task.end_time
    };
};

// Построение композиции для успешной задачи
const buildCompositionForTask = (task, links, tasks, taskIdToIndex, inAndOut) => {
    const stack = [task.id];
    const nodes = [];
    const localLinks = {};
    
    while (stack.length > 0) {
        const currentTaskId = stack.pop();
        const currentTask = tasks[taskIdToIndex[currentTaskId]];
        
        if (!currentTask) continue;
        
        const node = createCompositionNode(currentTask, inAndOut);
        if (!node) continue;
        
        // Обработка связей с этой задачей
        const taskLinks = links[currentTaskId.toString()];
        if (taskLinks) {
            taskLinks.forEach(link => {
                const sourceTask = tasks[taskIdToIndex[link.source]];
                if (!sourceTask) return;
                
                const linkKey = `${sourceTask.id}:${currentTask.id}`;
                
                if (localLinks[linkKey]) {
                    localLinks[linkKey].value.push(link.value);
            } else {
                    localLinks[linkKey] = {
                        source: sourceTask.id,
                        sourceMid: sourceTask.mid,
                        target: currentTask.id,
                        targetMid: currentTask.mid,
                        value: [link.value]
                    };
                }
                
                stack.push(link.source);
            });
        }
        
        // Добавление ссылочных входов
        Object.values(localLinks).forEach(link => {
            link.value.forEach(params => {
                const [sourceParamName, targetParamName] = params.split(':');
                node.inputs.push({
                    name: targetParamName,
                    value: `ref::${link.source}::${sourceParamName}`,
                    type: inAndOut[link.targetMid]?.input?.[targetParamName]
                });
            });
        });
        
        nodes.push(node);
    }
    
    return { nodes, localLinks };
};

// Нормализация композиции
const normalizeComposition = (nodes, localLinks) => {
    // Сортировка узлов по времени завершения
    nodes.sort((a, b) => new Date(a.end_time) - new Date(b.end_time));
    
    const taskIdToLocalId = {};
    let compositionId = '';
    
    // Присвоение локальных ID и построение ID композиции
    const normalizedNodes = nodes.map((node, index) => {
        node.id = `task/${index + 1}`;
        taskIdToLocalId[node.taskId] = node.id;
        
        if (compositionId) compositionId += '_';
        compositionId += node.taskId;
        
        // Обновление ссылочных входов
        node.inputs = node.inputs.map(input => {
            if (input.value && typeof input.value === 'string' && input.value.includes('ref::')) {
                const [ref, taskId, sourceParamName] = input.value.split('::');
                input.value = `${ref}::${taskIdToLocalId[taskId]}::${sourceParamName}`;
            }
            return input;
        });
        
        return node;
    });
    
    // Нормализация связей
    const normalizedLinks = Object.values(localLinks).map(link => ({
        ...link,
        source: taskIdToLocalId[link.source],
        target: taskIdToLocalId[link.target]
    }));
    
    return {
        id: compositionId,
        nodes: normalizedNodes,
        links: normalizedLinks
    };
};

const recover = async (req, res) => {
    try {
        console.log('Starting service composition recovery...');
        
        // Построение карты связей сервисов
        // inAndOut содержит информацию о том, какие параметры каждого сервиса
        // могут связывать его с другими сервисами (файлы) или внешними данными
        const inAndOut = await buildServiceConnectionMap();
        
        // Получение всех задач в хронологическом порядке
        console.log('Loading tasks...');
    const tasks = await models.Call.findAll({
        order: [["id", "ASC"]],
    });

        // Инициализация основных структур данных для анализа
        const compositions = []; // Массив найденных композиций сервисов
        
        /**
         * fileValueTracker - отслеживание файлов и их источников
         * Структура: {
         *   "путь_к_файлу": {
         *     value: taskId,     // ID задачи, которая создала этот файл
         *     name: "paramName"  // имя параметра через который файл был создан
         *   }
         * }
         */
        const fileValueTracker = {};
        
        /**
         * taskLinks - связи между задачами через файлы
         * Структура: {
         *   [targetTaskId]: [{
         *     source: sourceTaskId,
         *     target: targetTaskId,
         *     value: "sourceParam:targetParam"
         *   }]
         * }
         */
        const taskLinks = {};
        
        // Маппинг ID задачи к её индексу в массиве для быстрого поиска
        const taskIdToIndex = {};
        
        // Уникальные пользователи системы (владельцы задач)
        const users = {};
        
        // Построение индекса задач и сбор пользователей
        tasks.forEach((task, index) => {
            taskIdToIndex[task.id] = index;
            if (task.owner) {
                users[task.owner] = true;
            }
        });
        
        console.log(`Processing ${tasks.length} tasks...`);
        
        // Основной цикл анализа задач для поиска композиций
        // Проходим по всем задачам в хронологическом порядке
        for (const task of tasks) {
            // Парсим входные и выходные данные задачи
            const inputs = safeJsonParse(task.input, {}); // Входные параметры вызова сервиса
            const result = safeJsonParse(task.result, {}); // Результат выполнения задачи
            
            // Пропускаем задачи, для которых нет информации о сервисе
            if (!inAndOut[task.mid]) continue;
            
            // Получаем информацию о входных и выходных параметрах сервиса
            const { input: serviceInputs, output: serviceOutputs } = inAndOut[task.mid];
            
            // Определяем, является ли задача успешно завершенной с WMS-ссылкой
            // Такие задачи считаются конечными точками композиций
            const isSuccessfulWithWms = result && 
                result.status === 'success' && 
                result.hasOwnProperty('wms_link');
            
            if (isSuccessfulWithWms) {
                // === ОБРАБОТКА УСПЕШНЫХ ЗАДАЧ С WMS (конечные точки композиций) ===
                
                // Ищем связи этой задачи с предыдущими через файлы
                Object.keys(serviceInputs).forEach(paramName => {
                    const inputValue = inputs[paramName];
                    
                    // Если входной файл был создан другой задачей (и это не тематический выбор)
                    if (fileValueTracker[inputValue] && serviceInputs[paramName] !== WIDGET_TYPES.THEME_SELECT) {
                        const { value: sourceId, name: sourceParamName } = fileValueTracker[inputValue];
                        // Создаем связь между задачами
                        addTaskLink(taskLinks, task.id, sourceId, sourceParamName, paramName);
                    }
                });
                
                // Строим полную композицию, начиная от этой конечной задачи
                // Функция рекурсивно проходит по всем связанным задачам
                const { nodes, localLinks } = buildCompositionForTask(
                    task, taskLinks, tasks, taskIdToIndex, inAndOut
                );
                
                // Сохраняем композицию только если она содержит более одной задачи
                if (nodes.length > 1) {
                    const composition = normalizeComposition(nodes, localLinks);
                    compositions.push(composition);
                }
                        } else {
                // === ОБРАБОТКА ПРОМЕЖУТОЧНЫХ ЗАДАЧ ===
                
                // Для промежуточных задач только фиксируем связи, но не строим композиции
                Object.keys(serviceInputs).forEach(paramName => {
                    const inputValue = inputs[paramName];
                    
                    // Если входной файл был создан другой задачей
                    if (fileValueTracker[inputValue]) {
                        const { value: sourceId, name: sourceParamName } = fileValueTracker[inputValue];
                        // Фиксируем связь для будущего использования
                        addTaskLink(taskLinks, task.id, sourceId, sourceParamName, paramName);
                    }
                });
                
                // Регистрируем выходные файлы этой задачи для отслеживания
                if (result) {
                    Object.keys(serviceOutputs).forEach(paramName => {
                        const outputValue = result[paramName];
                        if (outputValue) {
                            // Запоминаем, что этот файл был создан данной задачей
                            fileValueTracker[outputValue] = {
                                value: task.id,    // ID задачи-создателя
                                name: paramName    // Имя выходного параметра
                            };
                        }
                        });
                    }
                }
        }
        
        console.log(`Created ${compositions.length} compositions`);
        
        // Сохранение результатов
        await createCompositions(compositions);
        await createUsers(users);
        
        if (res) {
            res.json({
                success: true,
                message: 'Service composition recovery completed',
                compositionsCount: compositions.length,
                usersCount: Object.keys(users).length
            });
        }
        
    } catch (error) {
        console.error('Error in recover function:', error);
        if (res) {
            res.status(500).json({
                success: false,
                error: 'Service composition recovery failed',
                message: error.message
            });
        }
    }
};

const fetchAllCompositions = async (req, res) => {
    try {
        // Используйте метод findAll, чтобы получить все композиции из базы данных
        const compositions = await models.Composition.findAll();

        // Отправьте полученные композиции в формате JSON в ответе
        console.log(compositions);
        res.send(compositions);
    } catch (error) {
        console.error("Ошибка при получении композиций:", error);
        res.status(500).json({ error: "Ошибка при получении композиций" });
    }
};

const getCompositionStats = async (req, res) => {
    let taskIdToIndex = {};
    let nodes = {};
    let links = {};
    const tasks = await models.Call.findAll({
        order: [["id", "DESC"]],
    });
    for (let i = 0; i < tasks.length; i++) {
        taskIdToIndex[tasks[i].id] = i;
        let inputs = JSON.parse(tasks[i].input);
        let result = JSON.parse(tasks[i].result);

        if (inputs.theme && inputs.theme.dataset_id) {
            let dataset_id = inputs.theme.dataset_id;
            let mid = tasks[i].mid;
            let owner = tasks[i].owner;
            if (nodes[dataset_id]) {
                if (nodes[dataset_id][owner]) {
                    nodes[dataset_id][owner]++;
                } else {
                    nodes[dataset_id][owner] = 1;
                }
            } else {
                nodes[dataset_id] = {
                    id: dataset_id,
                    [owner]: 1,
                };
            }
            if (nodes[mid]) {
                if (nodes[mid][owner]) {
                    nodes[mid][owner]++;
                } else {
                    nodes[mid][owner] = 1;
                }
            } else {
                nodes[mid] = {
                    id: mid,
                    [owner]: 1,
                };
            }
            if (links[`${dataset_id}:${mid}`]) {
                if (links[`${dataset_id}:${mid}`]["stats"][owner]) {
                    links[`${dataset_id}:${mid}`]["stats"][owner]++;
                } else {
                    links[`${dataset_id}:${mid}`]["stats"][owner] = 1;
                }
                links[`${dataset_id}:${mid}`]["stats"]["total"]++;
            } else {
                links[`${dataset_id}:${mid}`] = {
                    source: dataset_id,
                    target: mid,
                    stats: {
                        [owner]: 1,
                        total: 1,
                    },
                };
            }
        }
    }
    console.log("nodes", Object.keys(nodes).length);
    console.log("links", Object.keys(links).length);

    const compositions = await models.Composition.findAll();
    let path = {};
    for (let composition of compositions) {
        let composition_elements = composition.nodes;
        let lastTaskId =
            composition_elements[composition_elements.length - 1]["taskId"];
        let owner = tasks[taskIdToIndex[lastTaskId]].owner;
        let path_str = "";
        for (let i = 0; i < composition_elements.length; i++) {
            let node = composition_elements[i];
            let mid = node.mid;
            path_str = path_str + mid + ".";
            if (nodes[mid]) {
                if (nodes[mid][owner]) {
                    nodes[mid][owner]++;
                } else {
                    nodes[mid][owner] = 1;
                }
            } else {
                nodes[mid] = {
                    id: mid,
                    [owner]: 1,
                };
            }

            if (i < composition_elements.length - 1) {
                let sourceMid = composition_elements[i].mid;
                let targetMid = composition_elements[i + 1].mid;
                if (links[`${sourceMid}:${targetMid}`]) {
                    if (links[`${sourceMid}:${targetMid}`]["stats"][owner]) {
                        links[`${sourceMid}:${targetMid}`]["stats"][owner]++;
                    } else {
                        links[`${sourceMid}:${targetMid}`]["stats"][owner] = 1;
                    }
                    links[`${sourceMid}:${targetMid}`]["stats"]["total"]++;
                } else {
                    links[`${sourceMid}:${targetMid}`] = {
                        source: sourceMid,
                        target: targetMid,
                        stats: {
                            [owner]: 1,
                            total: 1,
                        },
                    };
                }
            }
        }
        path[path_str] = true;
    }
    console.log(path);
    console.log("nodes", Object.keys(nodes).length);
    console.log("links", Object.keys(links).length);
    let filePath = __dirname + "/statsGraph.json";
    let result = {
        nodes,
        links,
    };
    fs.writeFile(filePath, JSON.stringify(result), () => {
        console.log("write to statsGraph");
    });

    res.send(result);
};

// Обработка связи с датасетом
const processDatasetConnection = (call, inputParamName, inputValue, guidMap, datasetLinks, serviceDatasetEdges) => {
    try {
        // Парсим значение если это строка
        const parsedInput = typeof inputValue === 'string' ? 
            safeJsonParse(inputValue, inputValue) : inputValue;
            
        if (!parsedInput?.dataset_id) return;
        
        const normalizedDatasetId = normalizeDatasetId(parsedInput.dataset_id, guidMap);
        
        // Сохраняем связь между вызовом и датасетом
        datasetLinks[call.id] = `${normalizedDatasetId}:${inputParamName}`;
        
        // Обновляем статистику связей датасет-сервис
        updateServiceDatasetEdges(serviceDatasetEdges, normalizedDatasetId, call.mid, call.owner);
        
    } catch (error) {
        console.error('Error processing dataset connection:', error);
    }
};

// Обновление статистики связей датасет-сервис
const updateServiceDatasetEdges = (edges, datasetId, serviceId, owner) => {
    if (!edges[datasetId]) {
        edges[datasetId] = {};
    }
    
    if (!edges[datasetId][serviceId]) {
        edges[datasetId][serviceId] = { total: 0 };
    }
    
    if (!edges[datasetId][serviceId][owner]) {
        edges[datasetId][serviceId][owner] = 0;
    }
    
    edges[datasetId][serviceId][owner]++;
    edges[datasetId][serviceId].total++;
};

const recoverNew = async (req, res) => {
    try {
        console.log('Starting advanced service composition recovery...');
        
        // Параллельное построение карт связей для оптимизации производительности
        const [inAndOut, guidMap] = await Promise.all([
            buildServiceConnectionMap(), // Карта связей сервисов (входы/выходы)
            buildDatasetGuidMap()       // Маппинг GUID датасетов к их ID
        ]);
        
        // Получение всех вызовов сервисов в хронологическом порядке
        console.log('Loading calls...');
        const calls = await models.Call.findAll({
            order: [["id", "ASC"]],
        });

        // Инициализация основных структур данных для расширенного анализа
        
        /**
         * datasetLinks - связи между вызовами и датасетами
         * Структура: {
         *   [callId]: "datasetId:inputParamName"
         * }
         * Показывает, какой вызов использует какой датасет через какой параметр
         */
        const datasetLinks = {};
        
        /**
         * serviceDatasetEdges - статистика использования датасетов сервисами
         * Структура: {
         *   [datasetId]: {
         *     [serviceId]: {
         *       [ownerId]: количество_использований,
         *       total: общее_количество
         *     }
         *   }
         * }
         */
        const serviceDatasetEdges = {};
        
        /**
         * fileTracker - отслеживание файлов и их создателей
         * Структура: {
         *   "путь_к_файлу": {
         *     source_call_id: callId,      // ID вызова, создавшего файл
         *     source_service_id: serviceId, // ID сервиса, создавшего файл
         *     source_param_name: "paramName" // Имя выходного параметра
         *   }
         * }
         */
        const fileTracker = {};
        
        /**
         * callEdges - связи между вызовами через файлы
         * Структура: {
         *   [targetCallId]: {
         *     [sourceCallId]: ["sourceParam:targetParam", ...]
         *   }
         * }
         */
        const callEdges = {};
        
        // Индекс для быстрого поиска вызовов по ID
        const callIdToIndex = {};
        
        // Уникальные пользователи (владельцы вызовов)
        const users = {};
        
        // Построение индекса вызовов
        calls.forEach((call, index) => {
            callIdToIndex[call.id] = index;
            if (call.owner) {
                users[call.owner] = true;
            }
        });
        
        console.log(`Processing ${calls.length} calls...`);
        
        // === ПЕРВЫЙ ПРОХОД: Анализ связей между вызовами ===
        // Проходим по всем успешным вызовам и строим граф зависимостей
        calls.forEach((call, index) => {
            try {
                // Обрабатываем только успешно завершенные вызовы
                if (call.status !== TASK_STATUS.SUCCEEDED) return;
                
                // Парсим входные и выходные данные вызова
                const inputs = safeJsonParse(call.input, {}); // Параметры, переданные сервису
                const outputs = safeJsonParse(call.result, {}); // Результаты выполнения сервиса
                
                // Пропускаем вызовы неизвестных сервисов
                if (!inAndOut[call.mid]) return;
                
                // Получаем метаинформацию о параметрах сервиса
                const { input: serviceInputs, output: serviceOutputs } = inAndOut[call.mid];
                
                // Проверяем корректность метаданных сервиса
                if (!serviceInputs || !serviceOutputs) return;
                
                // === АНАЛИЗ ВХОДНЫХ ПАРАМЕТРОВ ===
                // Ищем связи с датасетами и файлами от других вызовов
                Object.keys(serviceInputs).forEach(paramName => {
                    try {
                        const inputValue = inputs[paramName];
                        if (!inputValue) return;
                        
                        // Определяем тип виджета (способ ввода данных)
                        const widgetType = serviceInputs[paramName];
                        
                        if (widgetType === WIDGET_TYPES.THEME_SELECT) {
                            // Обработка связи с датасетом (тематический выбор)
                            // Создает записи в datasetLinks и serviceDatasetEdges
                            processDatasetConnection(call, paramName, inputValue, guidMap, datasetLinks, serviceDatasetEdges);
                        } else if (widgetType === WIDGET_TYPES.FILE) {
                            // Обработка файловой связи между вызовами
                            const fileInfo = fileTracker[inputValue];
                            if (fileInfo && fileInfo.source_call_id && fileInfo.source_param_name) {
                                const { source_call_id, source_param_name } = fileInfo;
                                
                                // Создаем связь: источник -> текущий вызов
                                if (!callEdges[call.id]) callEdges[call.id] = {};
                                if (!callEdges[call.id][source_call_id]) callEdges[call.id][source_call_id] = [];
                                
                                // Сохраняем маппинг параметров: "выходной_параметр:входной_параметр"
                                callEdges[call.id][source_call_id].push(`${source_param_name}:${paramName}`);
                            }
                        }
                    } catch (paramError) {
                        console.error(`Error processing input param ${paramName} for call ${call.id}:`, paramError);
                    }
                });
                
                // === РЕГИСТРАЦИЯ ВЫХОДНЫХ ПАРАМЕТРОВ ===
                // Регистрируем файлы, созданные этим вызовом, для будущего использования
                Object.keys(serviceOutputs).forEach(paramName => {
                    try {
                        const outputValue = outputs[paramName];
                        
                        // Если параметр создает файл, регистрируем его в трекере
                        if (outputValue && serviceOutputs[paramName] === WIDGET_TYPES.FILE_SAVE) {
                            fileTracker[outputValue] = {
                                source_call_id: call.id,        // ID вызова-создателя
                                source_service_id: call.mid,    // ID сервиса-создателя
                                source_param_name: paramName    // Имя выходного параметра
                            };
                        }
                    } catch (paramError) {
                        console.error(`Error processing output param ${paramName} for call ${call.id}:`, paramError);
                    }
                });
                
            } catch (callError) {
                console.error(`Error processing call ${call.id} at index ${index}:`, callError);
            }
        });
        
        // === ВТОРОЙ ПРОХОД: Построение композиций ===
        // Создаем граф композиций на основе найденных связей
        
        /**
         * rawCompositions - промежуточные композиции
         * Структура: {
         *   [callId]: {
         *     nodes: [датасет/вызов, вызов, ...], // Узлы композиции в порядке выполнения
         *     links: [{source, target, fields}, ...] // Связи между узлами
         *   }
         * }
         */
        const rawCompositions = {};
        
        calls.forEach(call => {
            if (call.status !== TASK_STATUS.SUCCEEDED) return;
            
            // === ОБРАБОТКА СВЯЗЕЙ С ДАТАСЕТАМИ ===
            // Если вызов использует датасет, создаем начальную композицию
            if (datasetLinks[call.id]) {
                const [datasetId, paramName] = datasetLinks[call.id].split(':');
                
                // Создаем узел датасета (виртуальный узел)
                const datasetNode = { 
                    id: datasetId, 
                    start_date: call.start_date 
                };
                
                // Создаем связь датасет -> вызов
                const datasetLink = { 
                    source: datasetId, 
                    target: call.id, 
                    fields: `${datasetId}:${paramName}` 
                };
                
                // Инициализируем композицию с датасетом в качестве источника
                rawCompositions[call.id] = {
                    nodes: [datasetNode, call],
                    links: [datasetLink]
                };
            }
            
            // === ОБРАБОТКА ФАЙЛОВЫХ СВЯЗЕЙ МЕЖДУ ВЫЗОВАМИ ===
            // Расширяем композицию, добавляя связанные вызовы
            if (callEdges[call.id]) {
                Object.keys(callEdges[call.id]).forEach(sourceCallId => {
                    const fields = callEdges[call.id][sourceCallId]; // Маппинг параметров
                    const link = { source: sourceCallId, target: call.id, fields };
                    
                    if (rawCompositions[sourceCallId]) {
                        // НАСЛЕДОВАНИЕ: Берем всю композицию от источника и добавляем текущий вызов
                        rawCompositions[call.id] = {
                            nodes: [...rawCompositions[sourceCallId].nodes, call],
                            links: [...rawCompositions[sourceCallId].links, link]
                        };
                    } else {
                        // СОЗДАНИЕ НОВОЙ: Создаем простую композицию из двух вызовов
                        const sourceCall = calls[callIdToIndex[sourceCallId]];
                        if (sourceCall) {
                            rawCompositions[call.id] = {
                                nodes: [sourceCall, call],
                                links: [link]
                            };
                        }
                    }
                });
            }
        });
        
        // === ИЗВЛЕЧЕНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ ===
        // Создаем строковые представления композиций для анализа
        
        const callSequences = [];    // Последовательности ID вызовов
        const serviceSequences = []; // Последовательности ID сервисов
        
        Object.values(rawCompositions).forEach(composition => {
            const callIds = [];    // ID вызовов в композиции
            const serviceIds = []; // ID сервисов в композиции
            
            // Извлекаем ID из узлов композиции (пропускаем датасеты)
            composition.nodes.forEach(node => {
                if (node?.mid) { // Только узлы-вызовы (у датасетов нет mid)
                    callIds.push(node.id);
                    serviceIds.push(node.mid);
                }
            });
            
            // Создаем строковые представления для дальнейшего анализа
            if (callIds.length > 0) {
                callSequences.push(callIds.join('_'));      // "123_124_125"
                serviceSequences.push(serviceIds.join('_')); // "1_2_3"
            }
        });
        
        // === ФИЛЬТРАЦИЯ И ОПТИМИЗАЦИЯ ПОСЛЕДОВАТЕЛЬНОСТЕЙ ===
        
        /**
         * Фильтр для удаления префиксных последовательностей
         * Удаляет короткие последовательности, которые являются началом более длинных
         * Например: ["1_2", "1_2_3", "4_5"] -> ["1_2_3", "4_5"]
         */
        const filterNonPrefixSequences = (sequences) => {
            return sequences.filter((seq, i) => {
                // Проверяем, не является ли текущая последовательность префиксом другой
                return !sequences.some((other, j) => 
                    i !== j && other.startsWith(seq)
                );
            });
        };
        
        // Получаем самые длинные уникальные последовательности
        const longestCallSequences = filterNonPrefixSequences(callSequences);
        const longestServiceSequences = filterNonPrefixSequences([...new Set(serviceSequences)]);
        
        // === ПОСТРОЕНИЕ ФИНАЛЬНЫХ КОМПОЗИЦИЙ ===
        // Преобразуем отфильтрованные последовательности обратно в объекты композиций
        const finalCompositions = longestCallSequences.map(sequence => {
            const callIds = sequence.split('_');
            const lastCallId = callIds[callIds.length - 1]; // Последний вызов в цепочке
            
            // Возвращаем полную композицию, заканчивающуюся этим вызовом
            return rawCompositions[lastCallId];
        }).filter(Boolean); // Удаляем пустые результаты
        
        // === СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===
        // Записываем граф композиций в файл для дальнейшего анализа
        const outputPath = path.join(__dirname, "..", "compositionsDAG.json");
        fs.writeFile(outputPath, JSON.stringify(finalCompositions), (err) => {
            if (err) {
                console.error('Error writing compositionsDAG.json:', err);
                    } else {
                console.log("Compositions DAG saved successfully to:", outputPath);
            }
        });
        
        console.log(`Created ${finalCompositions.length} final compositions`);
        
        if (res) {
            res.json({
                success: true,
                message: 'Advanced composition recovery completed',
                longest_service_seq: longestServiceSequences,
                longest_comp: longestCallSequences,
                res_compositions: finalCompositions,
                stats: {
                    totalCalls: calls.length,
                    finalCompositions: finalCompositions.length,
                    users: Object.keys(users).length
                }
            });
        }
        
    } catch (error) {
        console.error('Error in recoverNew function:', error);
        if (res) {
            res.status(500).json({
                success: false,
                error: 'Advanced composition recovery failed',
                message: error.message
            });
        }
    }
};

module.exports = {
    recover,
    recoverNew,
    fetchAllCompositions,
    getCompositionStats,
};
