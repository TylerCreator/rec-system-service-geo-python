const datasetRouter = require('express').Router();
const {
  updateDatasets
} = require('../controllers/datasets');

datasetRouter.route('/update').get(updateDatasets);

module.exports = datasetRouter;