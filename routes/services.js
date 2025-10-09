const servicesRouter = require('express').Router();
const cors = require('cors');
const {
  getServices,
  updateServices,
  getRecomendations,
  getRecomendation,
  getServiceParameters,
  getPopularServices,
} = require('../controllers/services');

servicesRouter.route('/').get(getServices);
servicesRouter.route('/getRecomendations').get(cors(),getRecomendations);
servicesRouter.route('/getRecomendation').get(cors(),getRecomendation);
servicesRouter.route('/popular').get(cors(), getPopularServices);
servicesRouter.route('/parameters/:serviceId').get(cors(), getServiceParameters);

module.exports = servicesRouter;
