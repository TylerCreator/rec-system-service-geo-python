const updateRouter = require('express').Router();
const {
  updateAll,
  updateStatics,
  updateRecomendations,
  runFullUpdate
} = require('../controllers/update');

updateRouter.route('/all').get(updateAll);
updateRouter.route('/recomendations').get(updateRecomendations);
updateRouter.route('/statistic').get(updateStatics);

// Полное обновление системы (для cron job и ручного запуска)
updateRouter.route('/full').get(runFullUpdate);

// Простое обновление только статистики без внешних API
updateRouter.route('/local').get(async (req, res) => {
    try {
        console.log("Starting local update...");
        
        // Импортируем внутреннюю функцию
        const { updateStatisticsInternal } = require('../controllers/update');
        const result = await updateStatisticsInternal();
        
        res.status(200).json({
            message: "Local update completed successfully",
            result: result,
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.error("Error in local update:", error);
        res.status(500).json({
            error: "Local update failed", 
            message: error.message
        });
    }
});

module.exports = updateRouter;
