const axios = require("axios");
const models = require("../models/models"); // Путь к модели
const { recover } = require("./compositions");
const fs = require("fs");
const { Op } = require("sequelize");

const baseUrl =
    "http://cris.icc.ru/dataset/list?f=185&count_rows=true&iDisplayStart=0&iDisplayLength=";

const createBaseUrl = (displayStart, displayLength) => {
    return `http://cris.icc.ru/dataset/list?f=185&count_rows=true&unique=undefined&count_rows=1&iDisplayStart=${displayStart}&iDisplayLength=${
        displayStart + displayLength
    }`;
};

const getMinMaxId = async () => {
    try {
        const maxId = await models.Service.max("id");
        const minId = await models.Service.min("id");
        console.log("Max ID:", maxId);
        console.log("Min ID:", minId);
        return { minId, maxId };
    } catch (error) {
        console.error("Error fetching or saving data:", error);
    }
};
const updateServices = async (req, res) => {
    const requestData = {
        sort: [{ fieldname: "id", dir: false }],
    };

    let displayLength = 1;

    const serviceWithMaxIdInRemoteServer = await axios
        .post(`${baseUrl}${displayLength}`, requestData)
        .then((response) => response.data)
        .catch((err) => console.log(err));

    const minMaxIdInDataBase = await getMinMaxId();

    const totalRecords = +serviceWithMaxIdInRemoteServer.iTotalDisplayRecords;

    const serviceData = await models.Service.findAll();

    try {
        const displayLength = 100;
        let iDisplayStart = 0;
        let counter = 1;

        while (iDisplayStart < totalRecords) {
            console.log("services update counter", counter);
            const requestUrl = createBaseUrl(iDisplayStart, displayLength);

            const response = await axios.post(requestUrl, requestData);
            const data = response.data.aaData;

            if (data.length === 0) {
                console.log("services data пустая");
                break;
            }
            for (const item of data) {
                // // Проверяем, существует ли запись с таким же 'id' в базе данных
                // const existingService = serviceData.find((dbItem) => dbItem.id === item.id);

                // if (!existingService) {
                //   // Если записи нет, то добавляем ее в базу данных

                // }
                await models.Service.findOrCreate({
                    where: { id: item.id },
                    defaults: {
                        id: item.id,
                        name: item.name,
                        subject: item.subject,
                        type: item.type,
                        description: item.description,
                        actionview: item.actionview,
                        actionmodify: item.actionmodify,
                        map_reduce_specification: item.map_reduce_specification,
                        params: item.params,
                        js_body: item.js_body,
                        wpsservers: item.wpsservers,
                        wpsmethod: item.wpsmethod,
                        status: item.status,
                        output_params: item.output_params,
                        wms_link: item.wms_link,
                        wms_layer_name: item.wms_layer_name,
                        is_deleted: item.is_deleted,
                        created_by: item.created_by,
                        edited_by: item.edited_by,
                        edited_on: item.edited_on,
                        created_on: item.created_on,
                        classname: item.classname,
                    },
                });
            }

            iDisplayStart += displayLength;

            if (data.length < displayLength) {
                console.log("services закончились");
                break;
            }
        }

        console.log("Data synchronization completed.");
    } catch (error) {
        console.error("Error during data synchronization:", error);
        throw error;
    }
};

const getServices = async (req, res) => {
    console.log("getted services");
    let serviceData;
    try {
        if (req.query.user) {
            serviceData = await models.Service.findAll({
                subQuery: false,
                ...(req.query.limit && { limit: req.query.limit }),
                include: {
                    model: models.UserService,
                    where: {
                        user_id: req.query.user,
                    },
                    attributes: ["number_of_calls"],
                },
                order: [
                    [
                        {
                            model: models.UserService,
                        },
                        "number_of_calls",
                        "DESC",
                    ],
                ],
            });
        } else {
            await recover();
            serviceData = await models.Service.findAll({
                order: [["id", "DESC"]],
            });
        }

        res.send(serviceData);
        console.log("Service data from the database:", serviceData.length);
    } catch (error) {
        console.error("Error fetching data from the database:", error);
        throw error;
    }
};

//  function deleteAllServices() {
//     try {
//         // Удалить все записи из таблицы Service
//         await models.Service.destroy({
//             where: {},
//             truncate: true, // Очистить таблицу полностью
//         });

//         console.log("Все записи в таблице Service удалены.");
//     } catch (error) {
//         console.error("Ошибка при удалении записей из таблицы Service:", error);
//     }
// }

const getRecomendations = async (req, res) => {
    try {
        console.log("recomendations");
        const { spawn } = require("child_process");
        // const pythonProcess = spawn("python3", [
        //     "./recomendations/knn.py",
        //     "./../calls.csv",
        //     req.query.user_id,
        // ]);
        const pythonProcess = spawn("python3", [
            "knn.py",
            "./calls.csv",
            req.query.user_id,
        ]);
        console.log("recomendations 2");
        const answer = []
        pythonProcess.stdout.on("data", (data) => {
            console.log("data", answer.push(JSON.parse(data.toString())));
        });
        pythonProcess.stdout.on("end", (data) => {
            console.log("end", answer[0].prediction );
            res.send(answer[0])
        });
        
        console.log("recomendations 3");
    } catch (error) {
        console.error("Ошибка при создании рекомендации:", error);
    }
};


const getRecomendation = async (req, res) => {
    try {
        console.log("recomendation");
        
        let file = fs.readFileSync('recomendations.json', 'utf8');
        let recomendations = JSON.parse(file);
        if (req.query.user_id && recomendations['prediction'][req.query.user_id]) {
            res.send(recomendations?.prediction[req.query.user_id])
        } else{
            res.send([])
        } 
    } catch (error) {
        console.error("Ошибка при создании рекомендации:", error);
    }
};








/**
 * Получает список самых популярных сервисов среди пользователей
 * с фильтрами по типу сервиса и лимитом
 */
const getPopularServices = async (req, res) => {
    try {
        const {
            type = 'any', // 'table', 'dataset', 'service', 'any'
            limit = 20,
            period = 'all', // 'week', 'month', 'year', 'all'
            min_calls = 1, // минимальное количество вызовов для попадания в список
            user_id, // опциональный фильтр по пользователю
            ids_only = false // если true, возвращает только массив ID без дополнительных полей
        } = req.query;

        console.log(`Getting popular services: type=${type}, limit=${limit}, period=${period}, user_id=${user_id || 'all users'}`);

        // Строим условие для фильтрации по времени
        let timeCondition = {};
        if (period !== 'all') {
            const now = new Date();
            let startDate;
            
            switch (period) {
                case 'week':
                    startDate = new Date(now - 7 * 24 * 60 * 60 * 1000);
                    break;
                case 'month':
                    startDate = new Date(now - 30 * 24 * 60 * 60 * 1000);
                    break;
                case 'year':
                    startDate = new Date(now - 365 * 24 * 60 * 60 * 1000);
                    break;
            }
            
            if (startDate) {
                timeCondition.start_time = {
                    [Op.gte]: startDate
                };
            }
        }

        // Добавляем фильтр по пользователю, если указан
        let userCondition = {};
        if (user_id) {
            userCondition.owner = user_id;
        }

        // Получаем все успешные вызовы
        const calls = await models.Call.findAll({
            attributes: ['mid', 'owner'],
            where: {
                status: 'TASK_SUCCEEDED',
                ...timeCondition,
                ...userCondition
            }
        });

        // Группируем статистику вручную
        const serviceStats = {};
        calls.forEach(call => {
            if (!serviceStats[call.mid]) {
                serviceStats[call.mid] = {
                    mid: call.mid,
                    call_count: 0,
                    unique_users: new Set()
                };
            }
            serviceStats[call.mid].call_count++;
            serviceStats[call.mid].unique_users.add(call.owner);
        });

        // Преобразуем в массив и фильтруем по min_calls
        const callStats = Object.values(serviceStats)
            .filter(stat => stat.call_count >= parseInt(min_calls))
            .map(stat => ({
                mid: stat.mid,
                call_count: stat.call_count,
                unique_users: stat.unique_users.size
            }))
            .sort((a, b) => b.call_count - a.call_count)
            .slice(0, parseInt(limit) * 3); // Берем больше для фильтрации по типу

        // Разделяем ID на сервисы и таблицы
        const serviceIds = callStats.map(stat => stat.mid);
        const pureServiceIds = serviceIds.filter(id => id < 1000000);
        const tableIds = serviceIds.filter(id => id >= 1000000);

        let services = [];
        let datasets = [];

        // Получаем данные в зависимости от фильтра типа
        if (type === 'any' || type === 'service') {
            if (pureServiceIds.length > 0) {
                try {
                    services = await models.Service.findAll({
                        where: { id: { [Op.in]: pureServiceIds } }
                    });
                    console.log(`Found ${services.length} services`);
                } catch (error) {
                    console.error('Error fetching services:', error);
                    services = [];
                }
            }
        }

        if (type === 'any' || type === 'table' || type === 'dataset') {
            if (tableIds.length > 0) {
                try {
                    datasets = await models.Dataset.findAll({
                        where: { id: { [Op.in]: tableIds.map(id => id - 1000000) } },
                        attributes: ['id', 'guid'] // Используем только доступные поля
                    });
                    console.log(`Found ${datasets.length} datasets`);
                } catch (error) {
                    console.error('Error fetching datasets:', error);
                    datasets = [];
                }
            }
        }

        // Создаем единую карту для поиска
        const serviceMap = {};
        
        // Добавляем сервисы (с оригинальными ID)
        services.forEach(service => {
            const serviceData = service.toJSON ? service.toJSON() : service;
            serviceMap[service.id] = {
                id: serviceData.id,
                name: serviceData.name || `Service ${serviceData.id}`,
                type: serviceData.type || 'service',
                description: serviceData.description || '',
                subject: serviceData.subject || '',
                itemType: 'service'
            };
        });
        
        // Добавляем таблицы/датасеты (с ID + 1000000)
        datasets.forEach(dataset => {
            serviceMap[dataset.id + 1000000] = {
                id: dataset.id,
                name: dataset.guid || `Dataset ${dataset.id}`, // Используем guid как имя
                type: 'table',
                description: `Dataset with GUID: ${dataset.guid || 'Unknown'}`,
                subject: '',
                itemType: 'dataset'
            };
        });

        // Фильтруем и применяем лимит
        const filteredStats = callStats
            .filter(stat => serviceMap[stat.mid]) // Только те сервисы, которые прошли фильтр по типу
            .slice(0, parseInt(limit)); // Применяем лимит после фильтрации

        // Если нужны только ID, возвращаем простой массив
        if (ids_only === 'true' || ids_only === true) {
            const idsArray = filteredStats.map(stat => parseInt(stat.mid));
            return res.json(idsArray);
        }

        // Объединяем статистику с информацией о сервисах
        const popularServices = filteredStats.map(stat => {
            const item = serviceMap[stat.mid];
            const isDataset = stat.mid >= 1000000;
            
            return {
                itemId: parseInt(stat.mid),
                originalId: isDataset ? stat.mid - 1000000 : stat.mid, // Оригинальный ID без смещения
                itemName: item.name || 'Unknown Item',
                itemType: item.itemType || (isDataset ? 'dataset' : 'service'),
                serviceType: item.type || 'Unknown', // Сохраняем для обратной совместимости
                itemDescription: item.description || '',
                itemSubject: item.subject || '',
                callCount: parseInt(stat.call_count),
                uniqueUsers: parseInt(stat.unique_users),
                popularity: parseFloat((stat.call_count / Math.max(stat.unique_users, 1)).toFixed(2)),
                rank: callStats.findIndex(s => s.mid === stat.mid) + 1,
                // Для обратной совместимости
                serviceId: parseInt(stat.mid),
                serviceName: item.name || 'Unknown Item'
            };
        });

        // Получаем дополнительную статистику
        const totalServices = await models.Service.count();
        const totalDatasets = await models.Dataset.count();
        const totalCalls = calls.length;

        res.json({
            items: popularServices, // Переименовываем для ясности
            services: popularServices, // Оставляем для обратной совместимости
            meta: {
                total_services_in_db: totalServices,
                total_datasets_in_db: totalDatasets,
                total_items_in_db: totalServices + totalDatasets,
                total_successful_calls: totalCalls,
                filtered_by_type: type,
                time_period: period,
                filtered_by_user: user_id || null,
                user_specific: !!user_id,
                min_calls_threshold: parseInt(min_calls),
                limit: parseInt(limit),
                returned_count: popularServices.length,
                breakdown: {
                    services: popularServices.filter(item => item.itemType === 'service').length,
                    datasets: popularServices.filter(item => item.itemType === 'dataset').length
                },
                time_filter_applied: period !== 'all',
                generated_at: new Date().toISOString()
            }
        });

    } catch (error) {
        console.error("Ошибка при получении популярных сервисов:", error);
        res.status(500).json({
            error: 'Failed to get popular services',
            message: error.message
        });
    }
};

/**
 * Глубокий парсинг JSON строк, включая вложенные JSON
 * @param {string|object} data - Данные для парсинга
 * @returns {object} Распарсенный объект
 */
const deepParseJSON = (data) => {
    // Если уже объект, возвращаем как есть
    if (typeof data === 'object' && data !== null) {
        return data;
    }
    
    // Если не строка, возвращаем как есть
    if (typeof data !== 'string') {
        return data;
    }
    
    try {
        // Первый уровень парсинга
        const parsed = JSON.parse(data);
        
        // Если результат - объект, рекурсивно парсим все строковые значения
        if (typeof parsed === 'object' && parsed !== null) {
            const deepParsed = {};
            
            for (const [key, value] of Object.entries(parsed)) {
                if (typeof value === 'string') {
                    // Проверяем, является ли строка JSON
                    try {
                        const innerParsed = JSON.parse(value);
                        // Если успешно распарсили и это объект/массив, применяем рекурсивно
                        if (typeof innerParsed === 'object' && innerParsed !== null) {
                            deepParsed[key] = deepParseJSON(innerParsed);
                        } else {
                            deepParsed[key] = innerParsed;
                        }
                    } catch {
                        // Если не JSON, оставляем как строку
                        deepParsed[key] = value;
                    }
                } else if (typeof value === 'object' && value !== null) {
                    // Рекурсивно обрабатываем вложенные объекты
                    deepParsed[key] = deepParseJSON(value);
                } else {
                    deepParsed[key] = value;
                }
            }
            
            return deepParsed;
        }
        
        return parsed;
    } catch {
        // Если не удается распарсить, возвращаем исходные данные
        return data;
    }
};

/**
 * Получение возможных параметров сервиса с которыми он мог быть вызван
 * GET /services/parameters/:serviceId
 * Query params:
 * - user: фильтровать по конкретному пользователю (optional)
 * - limit: ограничить количество результатов (optional, default: 100)
 * - unique: вернуть только уникальные комбинации параметров (optional, default: true)
 */
const getServiceParameters = async (req, res) => {
    try {
        const { serviceId } = req.params;
        const { user, limit = 100, unique = 'true' } = req.query;

        console.log(`Getting parameters for service ${serviceId}`);

        // Валидация serviceId
        if (!serviceId || isNaN(parseInt(serviceId))) {
            return res.status(400).json({
                error: "Invalid service ID. Service ID must be a number.",
                serviceId
            });
        }

        // Проверяем существует ли сервис
        const service = await models.Service.findByPk(parseInt(serviceId));
        if (!service) {
            return res.status(404).json({
                error: "Service not found",
                serviceId: parseInt(serviceId)
            });
        }

        // Строим запрос для получения вызовов сервиса
        const whereCondition = {
            mid: parseInt(serviceId),
            input: {
                [models.Call.sequelize.Sequelize.Op.ne]: null
            }
        };

        // Добавляем фильтр по пользователю если указан
        if (user) {
            whereCondition.owner = user;
        }

        // Получаем вызовы сервиса с параметрами
        const serviceCalls = await models.Call.findAll({
            where: whereCondition,
            attributes: ['id', 'input', 'owner', 'start_time', 'status'],
            order: [['start_time', 'DESC']],
            limit: parseInt(limit) * 2 // Берем больше для фильтрации уникальных
        });

        if (serviceCalls.length === 0) {
            return res.json({
                service: {
                    id: service.id,
                    name: service.name,
                    description: service.description
                },
                parameters: [],
                schema: service.params ? JSON.parse(service.params) : null,
                totalCalls: 0,
                message: "No calls found for this service"
            });
        }

        // Обрабатываем параметры
        const processedParameters = [];
        const uniqueParameterHashes = new Set();

        for (const call of serviceCalls) {
            try {
                let inputParams = null;
                
                // Глубокий парсинг JSON параметров
                if (call.input) {
                    try {
                        inputParams = deepParseJSON(call.input);
                    } catch (parseError) {
                        console.warn(`Failed to parse input for call ${call.id}:`, parseError.message);
                        continue;
                    }
                }

                if (!inputParams) continue;

                // Создаем объект с параметрами
                const parameterSet = {
                    callId: call.id,
                    owner: call.owner,
                    timestamp: call.start_time,
                    status: call.status,
                    parameters: inputParams
                };

                // Если нужны уникальные параметры, проверяем хеш
                if (unique === 'true') {
                    const parameterHash = JSON.stringify(inputParams);
                    if (uniqueParameterHashes.has(parameterHash)) {
                        continue;
                    }
                    uniqueParameterHashes.add(parameterHash);
                }

                processedParameters.push(parameterSet);

                // Ограничиваем результат
                if (processedParameters.length >= parseInt(limit)) {
                    break;
                }
            } catch (error) {
                console.warn(`Error processing call ${call.id}:`, error.message);
                continue;
            }
        }

        // Анализируем типы параметров и их частоту
        const parameterAnalysis = analyzeParameters(processedParameters);
        
        // Анализируем популярные наборы параметров
        const parameterSetsAnalysis = analyzeParameterSets(processedParameters);

        // Подготавливаем ответ
        const response = {
            service: {
                id: service.id,
                name: service.name,
                description: service.description,
                type: service.type
            },
            parameters: processedParameters,
            analysis: parameterAnalysis,
            setsAnalysis: parameterSetsAnalysis,
            schema: service.params ? JSON.parse(service.params) : null,
            totalCalls: serviceCalls.length,
            returnedParameters: processedParameters.length,
            filters: {
                user: user || null,
                limit: parseInt(limit),
                unique: unique === 'true'
            }
        };

        res.json(response);

    } catch (error) {
        console.error("Error getting service parameters:", error);
        res.status(500).json({
            error: "Internal server error",
            message: error.message
        });
    }
};

/**
 * Анализирует параметры и возвращает статистику
 * Поддерживает подсчет популярных значений для всех типов данных:
 * - Простые типы (string, number, boolean) используются как есть
 * - Массивы и объекты преобразуются в JSON строки для подсчета
 * - Другие типы (null, undefined, function) преобразуются в строки
 * @param {Array} parameterSets - Массив наборов параметров для анализа
 * @returns {Object} Объект с анализом параметров
 */
const analyzeParameters = (parameterSets) => {
    const analysis = {
        parameterNames: {},
        parameterTypes: {},
        mostCommonValues: {},
        totalUniqueCombinations: parameterSets.length
    };

    for (const paramSet of parameterSets) {
        const params = paramSet.parameters;
        
        for (const [key, value] of Object.entries(params)) {
            // Подсчет имен параметров
            analysis.parameterNames[key] = (analysis.parameterNames[key] || 0) + 1;
            
            // Анализ типов
            const valueType = Array.isArray(value) ? 'array' : typeof value;
            if (!analysis.parameterTypes[key]) {
                analysis.parameterTypes[key] = {};
            }
            analysis.parameterTypes[key][valueType] = (analysis.parameterTypes[key][valueType] || 0) + 1;
            
            // Подсчет популярных значений (для всех типов данных)
            if (!analysis.mostCommonValues[key]) {
                analysis.mostCommonValues[key] = {};
            }
            
            let valueForCounting;
            if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
                // Простые типы используем как есть
                valueForCounting = value;
            } else if (Array.isArray(value) || typeof value === 'object') {
                // Массивы и объекты преобразуем к строке для подсчета
                try {
                    valueForCounting = JSON.stringify(value);
                } catch (stringifyError) {
                    // Если не удается сериализовать (например, циклические ссылки), пропускаем
                    console.warn(`Failed to stringify value for parameter ${key}:`, stringifyError.message);
                    continue;
                }
            } else {
                // Для других типов (null, undefined, function и т.д.) используем строковое представление
                valueForCounting = String(value);
            }
            
            analysis.mostCommonValues[key][valueForCounting] = (analysis.mostCommonValues[key][valueForCounting] || 0) + 1;
        }
    }

    // Сортируем популярные значения
    for (const [key, values] of Object.entries(analysis.mostCommonValues)) {
        analysis.mostCommonValues[key] = Object.entries(values)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 10) // Топ 10
            .reduce((obj, [val, count]) => {
                obj[val] = count;
                return obj;
            }, {});
    }

    return analysis;
};

/**
 * Анализирует популярные уникальные наборы параметров по parameterHash
 * @param {Array} parameterSets - Массив наборов параметров для анализа
 * @returns {Object} Объект с анализом популярных наборов параметров
 */
const analyzeParameterSets = (parameterSets) => {
    const analysis = {
        totalSets: parameterSets.length,
        uniqueSets: 0,
        popularSets: {},
        setFrequency: {},
        mostPopularParameters: {},
        setsByOwner: {}
    };

    // Группируем наборы параметров по хешу
    const parameterHashGroups = {};
    const ownerStats = {};

    for (const paramSet of parameterSets) {
        const parameterHash = JSON.stringify(paramSet.parameters);
        const owner = paramSet.owner;

        // Подсчет по хешам
        if (!parameterHashGroups[parameterHash]) {
            parameterHashGroups[parameterHash] = {
                count: 0,
                parameters: paramSet.parameters,
                owners: new Set(),
                firstUsed: paramSet.timestamp,
                lastUsed: paramSet.timestamp,
                callIds: []
            };
        }

        parameterHashGroups[parameterHash].count++;
        parameterHashGroups[parameterHash].owners.add(owner);
        parameterHashGroups[parameterHash].callIds.push(paramSet.callId);
        
        // Обновляем временные метки
        if (paramSet.timestamp < parameterHashGroups[parameterHash].firstUsed) {
            parameterHashGroups[parameterHash].firstUsed = paramSet.timestamp;
        }
        if (paramSet.timestamp > parameterHashGroups[parameterHash].lastUsed) {
            parameterHashGroups[parameterHash].lastUsed = paramSet.timestamp;
        }

        // Статистика по владельцам
        if (!ownerStats[owner]) {
            ownerStats[owner] = {
                totalCalls: 0,
                uniqueSets: new Set(),
                mostUsedSet: { hash: null, count: 0 }
            };
        }
        ownerStats[owner].totalCalls++;
        ownerStats[owner].uniqueSets.add(parameterHash);
    }

    // Подсчет уникальных наборов
    analysis.uniqueSets = Object.keys(parameterHashGroups).length;

    // Создание частотной статистики
    for (const [hash, data] of Object.entries(parameterHashGroups)) {
        analysis.setFrequency[hash] = data.count;
        
        // Обновляем статистику владельцев
        for (const owner of data.owners) {
            if (data.count > ownerStats[owner].mostUsedSet.count) {
                ownerStats[owner].mostUsedSet = {
                    hash: hash,
                    count: data.count,
                    parameters: data.parameters
                };
            }
        }
    }

    // Сортируем наборы по популярности
    const sortedSets = Object.entries(parameterHashGroups)
        .sort(([,a], [,b]) => b.count - a.count)
        .slice(0, 20); // Топ 20 популярных наборов

    // Формируем результат популярных наборов
    for (const [hash, data] of sortedSets) {
        analysis.popularSets[hash] = {
            count: data.count,
            parameters: data.parameters,
            uniqueOwners: data.owners.size,
            owners: Array.from(data.owners),
            firstUsed: data.firstUsed,
            lastUsed: data.lastUsed,
            totalCalls: data.callIds.length,
            callIds: data.callIds.slice(0, 10) // Первые 10 ID вызовов для примера
        };
    }

    // Анализ самых популярных параметров в наборах
    const parameterPopularity = {};
    for (const [hash, data] of Object.entries(parameterHashGroups)) {
        for (const [paramName, paramValue] of Object.entries(data.parameters)) {
            if (!parameterPopularity[paramName]) {
                parameterPopularity[paramName] = {};
            }
            
            const valueKey = typeof paramValue === 'object' ? 
                JSON.stringify(paramValue) : String(paramValue);
            
            if (!parameterPopularity[paramName][valueKey]) {
                parameterPopularity[paramName][valueKey] = 0;
            }
            parameterPopularity[paramName][valueKey] += data.count;
        }
    }

    // Топ популярных значений параметров
    for (const [paramName, values] of Object.entries(parameterPopularity)) {
        analysis.mostPopularParameters[paramName] = Object.entries(values)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 5) // Топ 5 для каждого параметра
            .reduce((obj, [val, count]) => {
                obj[val] = count;
                return obj;
            }, {});
    }

    // Статистика по владельцам
    for (const [owner, stats] of Object.entries(ownerStats)) {
        analysis.setsByOwner[owner] = {
            totalCalls: stats.totalCalls,
            uniqueSets: stats.uniqueSets.size,
            mostUsedSet: stats.mostUsedSet,
            diversityRatio: stats.uniqueSets.size / stats.totalCalls // Коэффициент разнообразия
        };
    }

    return analysis;
};

module.exports = {
    updateServices,
    getServices,
    getRecomendation,
    getRecomendations,
    getServiceParameters,
    getPopularServices,
};
