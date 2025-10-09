/* eslint-disable linebreak-style */
require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const nocache = require('nocache');
const cron = require('node-cron');
const morgan = require('morgan');
const fs = require('fs');
const https = require('https');
const http = require('http');
const waitPort = require('wait-port');

const {
  NODE_DOCKER_PORT: PORT = 8080,
} = process.env;

const sequelize = require('./db');
const callsRouter = require('./routes/calls.js');
const servicesRouter = require('./routes/services.js');
const updateRouter = require('./routes/update.js')
const compositionsRouter = require('./routes/compositions.js')
const {
  runFullUpdate
} = require('./controllers/update');
const datasetRouter = require('./routes/datasets.js');
const config = require("./config/config.json");

const app = express();

app.use(cors());
app.options('*', cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

app.use(morgan('dev'));

app.use(nocache());
app.set('etag', false);

app.get('/', (req, res)=> {
    console.log('getted')
    res.status(200).send({message: 'hello'})
})

app.use('/calls', callsRouter);
app.use('/datasets', datasetRouter);
app.use('/services', servicesRouter);
app.use('/update', updateRouter );
app.use('/compositions', compositionsRouter );

// Endpoint для ручного запуска cron задачи (для тестирования)
// Теперь используем централизованную функцию через /update/full
app.get('/admin/run-cron', (req, res) => {
  console.log('🔧 Manual cron job triggered via /admin/run-cron - redirecting to /update/full');
  res.redirect('/update/full');
});

app.use('/:404', (req, res, next) => {
  res.status(404).send({ message: 'страница не найдена' });
  next();
});

// axios.defaults.timeout = 30000;
// axios.defaults.httpsAgent = new https.Agent({ keepAlive: true });


const start = async () => {
  try {
      await waitPort({ host: process.env.DB_HOST, port: parseInt(process.env.DB_PORT,10), timeout: 60000 });

      await sequelize.authenticate()
      await sequelize.sync()
      // Проверка наличия сертификатов
      console.log(config, fs.existsSync(config.certPath))
      if (config && fs.existsSync(config.certPath) && fs.existsSync(config.keyPath)) {
        
          
        const options = {
          key: fs.readFileSync(config.keyPath),
          cert: fs.readFileSync(config.certPath),
        };


        https.createServer(options, app).listen(PORT,'0.0.0.0', () => {
          console.log(`HTTPS сервер работает на порту  ${PORT}`);
        });
      } else {
        http.createServer(app).listen(PORT,'0.0.0.0', () => {
          console.log(`HTTP сервер работает на порту  ${PORT}`);
        });
      }

      // Настраиваем cron задачу для ежедневного обновления в полночь по Иркутску
      cron.schedule('0 0 * * *', async () => {
        console.log('⏰ Daily cron job triggered');
        try {
          await runFullUpdate(); // Используем централизованную функцию
        } catch (error) {
          console.error('💥 Critical error in daily cron job:', error);
        }
      }, {
        scheduled: true,
        timezone: "Asia/Irkutsk"
      });

      console.log('⏰ Cron job scheduled: Daily at 00:00 Asia/Irkutsk timezone');
  } catch (e) {
      console.log(e)
  }
}

start()