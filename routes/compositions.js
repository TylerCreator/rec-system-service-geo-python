const compositionsRouter = require('express').Router();
const {
  recover,
  fetchAllCompositions,
  getCompositionStats,
  recoverNew
} = require('../controllers/compositions');

compositionsRouter.route('/recover').get(recover);
compositionsRouter.route('/recoverNew').get(recoverNew);
compositionsRouter.route('/').get(fetchAllCompositions);
compositionsRouter.route('/stats').get(getCompositionStats)



module.exports = compositionsRouter;
