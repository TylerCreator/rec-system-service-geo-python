const { updateCalls, dumpCsv } = require("./calls");
const { recover } = require("./compositions");
const servicesController = require("./services");
const models = require("../models/models"); // Путь к модели
const { updateDatasets } = require("./datasets");


const updateAll = async (req, res) => {
    const results = [];
    const isHttpCall = req && res;
    
    try {
        console.log("Starting updateAll process...");
        
        // 1. Обновление вызовов
        try {
            console.log("Step 1: Updating calls...");
            await updateCalls();
            results.push("✅ Calls updated successfully");
        } catch (error) {
            console.error("Error updating calls:", error.message);
            results.push(`❌ Calls update failed: ${error.message}`);
        }

        // 2. Обновление датасетов
        try {
            console.log("Step 2: Updating datasets...");
            await updateDatasets();
            results.push("✅ Datasets updated successfully");
        } catch (error) {
            console.error("Error updating datasets:", error.message);
            results.push(`❌ Datasets update failed: ${error.message}`);
        }

        // 3. Обновление сервисов
        try {
            console.log("Step 3: Updating services...");
            await servicesController.updateServices();
            results.push("✅ Services updated successfully");
        } catch (error) {
            console.error("Error updating services:", error.message);
            results.push(`❌ Services update failed: ${error.message}`);
        }

        // 4. Восстановление композиций
        try {
            console.log("Step 4: Recovering compositions...");
            await recover();
            results.push("✅ Compositions recovered successfully");
        } catch (error) {
            console.error("Error recovering compositions:", error.message);
            results.push(`❌ Compositions recovery failed: ${error.message}`);
        }

        console.log("UpdateAll process completed");

        // Если вызвана через HTTP, отправляем ответ
        if (isHttpCall) {
            res.status(200).json({
                message: "Update process completed",
                results: results,
                timestamp: new Date().toISOString()
            });
        }
        
        // Возвращаем результаты для программного использования
        return results;
        
    } catch (error) {
        console.error("Fatal error in updateAll:", error);
        
        // Если вызвана через HTTP, отправляем ошибку
        if (isHttpCall) {
            res.status(500).json({
                error: "Fatal error in update process",
                message: error.message,
                results: results
            });
        }
        
        // Пробрасываем ошибку для программного использования
        throw error;
    }
};

// Внутренняя функция для обновления статистики
const updateStatisticsInternal = async () => {
    console.log("Updating statistics internally...");
    
    const transaction = await models.Call.sequelize.transaction();
    
    try {
        // Используем SQL агрегацию через модель
        const [callStats] = await models.Call.sequelize.query(`
            SELECT owner, mid, COUNT(*) as call_count 
            FROM "Calls" 
            WHERE owner IS NOT NULL AND mid IS NOT NULL 
            GROUP BY owner, mid
        `, { transaction });

        console.log(`Processing ${callStats.length} user-service combinations`);

        if (callStats.length === 0) {
            console.log("No call statistics found");
            await transaction.commit();
            return "No call statistics to process";
        }

        // Создаем пользователей из статистики вызовов
        const uniqueOwners = [...new Set(callStats.map(stat => stat.owner))];
        const uniqueServiceIds = [...new Set(callStats.map(stat => stat.mid))];
        
        console.log(`Found ${uniqueOwners.length} unique users and ${uniqueServiceIds.length} unique services`);

        // Проверяем, какие сервисы существуют в базе данных
        const existingServices = await models.Service.findAll({
            where: {
                id: uniqueServiceIds
            },
            attributes: ['id'],
            transaction
        });
        
        const existingServiceIds = new Set(existingServices.map(service => service.id));
        console.log(`Found ${existingServices.length} existing services in database`);

        // Создаем пользователей пакетно
        const userCreationPromises = uniqueOwners.map(owner => 
            models.User.findOrCreate({
                where: { id: owner },
                defaults: { id: owner },
                transaction
            })
        );
        
        await Promise.all(userCreationPromises);
        console.log("Users created/found successfully");

        // Обновляем статистику пользователь-сервис только для существующих сервисов
        let processedCount = 0;
        let skippedCount = 0;
        
        for (const stat of callStats) {
            // Проверяем, существует ли сервис
            if (!existingServiceIds.has(parseInt(stat.mid))) {
                console.warn(`Service with id ${stat.mid} not found, skipping...`);
                skippedCount++;
                continue;
            }

            const callCount = parseInt(stat.call_count);
            
            // Используем upsert для обновления или создания записи
            const [userService, created] = await models.UserService.findOrCreate({
                where: {
                    user_id: stat.owner,
                    service_id: parseInt(stat.mid),
                },
                defaults: {
                    user_id: stat.owner,
                    service_id: parseInt(stat.mid),
                    number_of_calls: callCount,
                },
                transaction
            });

            // Если запись уже существовала, обновляем количество вызовов
            if (!created && userService.number_of_calls !== callCount) {
                await userService.update({
                    number_of_calls: callCount
                }, { transaction });
                console.log(`Updated calls count for user ${stat.owner} and service ${stat.mid}: ${callCount}`);
            }
            
            processedCount++;
        }

        await transaction.commit();
        console.log("Statistics updated successfully");
        
        const result = `Processed ${processedCount} user-service combinations for ${uniqueOwners.length} users. Skipped ${skippedCount} combinations due to missing services.`;
        console.log(result);
        return result;
        
    } catch (error) {
        await transaction.rollback();
        console.error("Error in updateStatisticsInternal:", error);
        throw new Error(`Failed to update statistics: ${error.message}`);
    }
};

const updateStatics = async (req, res) => {
    console.log("Starting statistics update...");
    try {
        const result = await updateStatisticsInternal();
        console.log("Statistics update completed successfully");
        
        res.status(200).json({
            success: true,
            message: "Statistics updated successfully",
            result: result,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        console.error("Error updating statistics:", error);
        
        res.status(500).json({
            success: false,
            error: "Failed to update statistics",
            message: error.message,
            timestamp: new Date().toISOString()
        });
    }
};

const updateRecomendations = async (req, res) => {
    try {
        console.log("Starting recommendations update...");
        
        const { spawn } = require("child_process");
        const fs = require('fs');
        
        // Проверяем и создаем файл recomendations.json с правильными правами доступа
        const recommendationsFile = 'recomendations.json';
        const tempFile = 'recomendations_temp.json';
        
        // Сначала создаем временный файл для проверки прав
        try {
            fs.writeFileSync(tempFile, '{"prediction": {}}', 'utf8');
            console.log("Temporary file created successfully");
            
            // Если существует старый файл, пытаемся его удалить
            if (fs.existsSync(recommendationsFile)) {
                try {
                    fs.unlinkSync(recommendationsFile);
                    console.log("Old recommendations file removed");
                } catch (unlinkError) {
                    console.warn("Could not remove old file, will use temp file approach");
                }
            }
            
            // Переименовываем временный файл
            fs.renameSync(tempFile, recommendationsFile);
            console.log("Recommendations file prepared successfully");
            
        } catch (error) {
            console.warn("Could not create recommendations file, Python script will handle it:", error.message);
        }

        const pythonProcess = spawn("python3", [
            "knn.py",
            "./calls.csv",
            req.query.user_id || "",
        ]);

        let outputData = '';
        let errorData = '';
        let hasResponded = false;

        // Обрабатываем вывод Python-скрипта
        pythonProcess.stdout.on("data", (data) => {
            outputData += data.toString();
            console.log("Python stdout chunk received");
        });

        // Обрабатываем ошибки Python-скрипта
        pythonProcess.stderr.on("data", (data) => {
            errorData += data.toString();
            console.error("Python stderr:", data.toString());
        });

        // Обрабатываем завершение процесса
        pythonProcess.on("close", (code) => {
            if (hasResponded) return;
            hasResponded = true;

            console.log(`Python process exited with code ${code}`);
            
            if (code !== 0) {
                console.error("Python process failed:", errorData);
                return res.status(500).json({
                    success: false,
                    error: "Failed to update recommendations",
                    message: `Python process exited with code ${code}`,
                    details: errorData.substring(0, 1000), // Ограничиваем размер ошибки
                    timestamp: new Date().toISOString()
                });
            }

            try {
                // Пытаемся парсить JSON из вывода Python-скрипта
                let result;
                if (outputData.trim()) {
                    result = JSON.parse(outputData);
                    console.log("Recommendations updated successfully");
                } else {
                    // Если нет вывода, читаем из файла
                    const fileContent = fs.readFileSync(recommendationsFile, 'utf8');
                    result = JSON.parse(fileContent);
                    console.log("Recommendations read from file successfully");
                }
                
                res.json({
                    success: true,
                    message: "Recommendations updated successfully",
                    data: result,
                    timestamp: new Date().toISOString()
                });
            } catch (parseError) {
                console.error("Failed to parse recommendations data:", parseError);
                res.status(500).json({
                    success: false,
                    error: "Failed to parse recommendations data",
                    message: parseError.message,
                    timestamp: new Date().toISOString()
                });
            }
        });

        // Обрабатываем ошибки запуска процесса
        pythonProcess.on("error", (error) => {
            if (hasResponded) return;
            hasResponded = true;
            
            console.error("Failed to start Python process:", error);
            res.status(500).json({
                success: false,
                error: "Failed to start recommendation update process",
                message: error.message,
                timestamp: new Date().toISOString()
            });
        });

        // Устанавливаем таймаут для процесса (60 секунд для обновления)
        setTimeout(() => {
            if (hasResponded) return;
            hasResponded = true;
            
            pythonProcess.kill('SIGTERM');
            console.error("Python process timeout");
            res.status(408).json({
                success: false,
                error: "Recommendation update timeout",
                message: "Process took too long to complete",
                timestamp: new Date().toISOString()
            });
        }, 60000);

    } catch (error) {
        console.error("Error in updateRecomendations:", error);
        res.status(500).json({
            success: false,
            error: "Internal server error",
            message: error.message,
            timestamp: new Date().toISOString()
        });
    }
};

// Полное обновление системы (для cron job и ручного запуска)
const runFullUpdate = async (req, res) => {
    console.log('🕐 Starting full system update at:', new Date().toISOString());
    const startTime = Date.now();
    const results = [];
    
    try {
        // 1. Обновление всех данных (вызовы, датасеты, сервисы, композиции)
        console.log('📊 Step 1: Updating all data...');
        try {
            // Вызываем updateAll без HTTP-контекста
            const mainResults = await updateAll();
            
            // Добавляем результаты основных операций
            results.push(...mainResults);
            
            // Проверяем, есть ли ошибки в основных операциях
            const hasMainErrors = mainResults.some(r => r.includes('❌'));
            if (hasMainErrors) {
                results.push('❌ updateAll: COMPLETED WITH ERRORS');
                console.log('⚠️ updateAll completed with some errors');
            } else {
                results.push('✅ updateAll: SUCCESS');
                console.log('✅ updateAll completed successfully');
            }
        } catch (error) {
            results.push(`❌ updateAll: FAILED - ${error.message}`);
            console.error('❌ updateAll failed:', error);
        }

        // 2. Дамп CSV файла (должен быть после updateAll)
        console.log('📄 Step 2: Dumping CSV...');
        try {
            await dumpCsv();
            results.push('✅ dumpCsv: SUCCESS');
            console.log('✅ dumpCsv completed successfully');
        } catch (error) {
            results.push(`❌ dumpCsv: FAILED - ${error.message}`);
            console.error('❌ dumpCsv failed:', error);
        }

        // 3. Обновление статистики пользователь-сервис (должно быть после dumpCsv)
        console.log('📈 Step 3: Updating statistics...');
        try {
            if (req && res) {
                // HTTP вызов
                await new Promise((resolve, reject) => {
                    const mockReq = {};
                    const mockRes = {
                        status: () => mockRes,
                        json: (data) => {
                            if (data.success) {
                                resolve(data);
                            } else {
                                reject(new Error(data.message || 'Statistics update failed'));
                            }
                        }
                    };
                    updateStatics(mockReq, mockRes).catch(reject);
                });
            } else {
                // Cron вызов
                await updateStatisticsInternal();
            }
            results.push('✅ updateStatics: SUCCESS');
            console.log('✅ updateStatics completed successfully');
        } catch (error) {
            results.push(`❌ updateStatics: FAILED - ${error.message}`);
            console.error('❌ updateStatics failed:', error);
        }

        // 4. Обновление рекомендаций (должно быть после updateStatics)
        console.log('🤖 Step 4: Updating recommendations...');
        try {
            if (req && res) {
                // HTTP вызов
                await new Promise((resolve, reject) => {
                    const mockReq = { query: {} };
                    const mockRes = {
                        status: () => mockRes,
                        json: (data) => {
                            if (data.success) {
                                resolve(data);
                            } else {
                                reject(new Error(data.message || 'Recommendations update failed'));
                            }
                        }
                    };
                    updateRecomendations(mockReq, mockRes).catch(reject);
                });
            } else {
                // Cron вызов - создаем минимальный процесс без HTTP ответа
                await new Promise((resolve, reject) => {
                    const mockReq = { query: {} };
                    const mockRes = {
                        status: () => mockRes,
                        json: (data) => {
                            if (data.success) {
                                resolve(data);
                            } else {
                                reject(new Error(data.message || 'Recommendations update failed'));
                            }
                        }
                    };
                    updateRecomendations(mockReq, mockRes).catch(reject);
                });
            }
            results.push('✅ updateRecomendations: SUCCESS');
            console.log('✅ updateRecomendations completed successfully');
        } catch (error) {
            results.push(`❌ updateRecomendations: FAILED - ${error.message}`);
            console.error('❌ updateRecomendations failed:', error);
        }

    } catch (criticalError) {
        console.error('💥 Critical error in full update:', criticalError);
        results.push(`💥 CRITICAL ERROR: ${criticalError.message}`);
    }

    const endTime = Date.now();
    const duration = ((endTime - startTime) / 1000 / 60).toFixed(2); // в минутах
    
    console.log('🏁 Full system update completed');
    console.log(`⏱️  Total execution time: ${duration} minutes`);
    console.log('📋 Results summary:', results);
    console.log('='.repeat(80));

    // Если это HTTP запрос, отправляем ответ
    if (req && res) {
        const hasErrors = results.some(result => result.includes('❌') || result.includes('💥'));
        
        res.status(hasErrors ? 207 : 200).json({
            success: !results.some(result => result.includes('💥')), // true если нет критических ошибок
            message: 'Full system update completed',
            executionTime: `${duration} minutes`,
            results: results,
            hasErrors: hasErrors,
            timestamp: new Date().toISOString()
        });
    }

    return {
        success: !results.some(result => result.includes('💥')),
        executionTime: duration,
        results: results
    };
};

module.exports = {
    updateAll,
    updateStatics,
    updateRecomendations,
    updateStatisticsInternal,
    runFullUpdate
};
