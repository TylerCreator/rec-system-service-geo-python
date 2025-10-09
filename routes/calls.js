const callsRouter = require('express').Router();
const {
  getCalls, incrCalls, updateCalls, dumpCsv
} = require('../controllers/calls');

callsRouter.route('/').get(getCalls);
callsRouter.route('/incr').get(incrCalls);
callsRouter.route('/update-calls').get(updateCalls);
callsRouter.route('/dump-csv').get(dumpCsv);

module.exports = callsRouter;
