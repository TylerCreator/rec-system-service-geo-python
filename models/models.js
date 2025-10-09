const sequelize = require('../db');
const {DataTypes} = require('sequelize');

const Call = sequelize.define('Call', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    unique: true
  },
  classname: DataTypes.STRING,
  console_output: DataTypes.TEXT,
  created_by: DataTypes.STRING,
  created_on: DataTypes.DATE,
  edited_by: DataTypes.STRING,
  edited_on: DataTypes.DATE,
  end_time: DataTypes.DATE,
  error_output: DataTypes.TEXT,
  input: DataTypes.TEXT,
  input_data: DataTypes.TEXT,
  input_params: DataTypes.TEXT,
  is_deleted: DataTypes.STRING,
  mid: DataTypes.INTEGER,
  os_pid: DataTypes.INTEGER,
  owner: DataTypes.STRING,
  result: DataTypes.TEXT,
  start_time: DataTypes.DATE,
  status: DataTypes.STRING,
});

const Service = sequelize.define('Service', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    unique: true
  },
  name: DataTypes.STRING,
  subject: DataTypes.STRING,
  type: DataTypes.STRING,
  description: DataTypes.TEXT,
  number_of_calls: DataTypes.INTEGER,
  actionview: DataTypes.STRING,
  actionmodify: DataTypes.STRING,
  map_reduce_specification: DataTypes.STRING,
  params: DataTypes.JSON,
  js_body: DataTypes.TEXT,
  wpsservers: DataTypes.JSON,
  wpsmethod: DataTypes.STRING,
  status: DataTypes.STRING,
  output_params: DataTypes.JSON,
  wms_link: DataTypes.STRING,
  wms_layer_name: DataTypes.STRING,
  is_deleted: DataTypes.STRING,
  created_by: DataTypes.STRING,
  edited_by: DataTypes.STRING,
  edited_on: DataTypes.DATE,
  created_on: DataTypes.DATE,
  classname: DataTypes.STRING,
});

const Composition = sequelize.define('Composition', {
  id: {
    type: DataTypes.TEXT,
    primaryKey: true,
    unique: true
  },
  // Определение полей модели Composition
  nodes: {
    type: DataTypes.JSON, // Массив объектов nodes
    allowNull: false,
  },
  links: {
    type: DataTypes.JSON, // Массив объектов links
    allowNull: false,
  },
});

const Dataset = sequelize.define('Datasets', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    unique: true
  },
  guid: {
    type: DataTypes.STRING,
  },
})


const User = sequelize.define('User', {
  id: {
    type: DataTypes.STRING,
    primaryKey: true,
    unique: true
  },
})

const UserService = sequelize.define('UserService', {
  number_of_calls: DataTypes.INTEGER
});

User.belongsToMany(Service, { through: UserService, foreignKey: 'user_id', otherKey: 'service_id' });
Service.belongsToMany(User, { through: UserService, foreignKey: 'service_id', otherKey: 'user_id' });

UserService.belongsTo(User, {
  foreignKey: {
    name: 'user_id'
  }
});
User.hasMany(UserService, {
  foreignKey: {
    name: 'user_id'
  }
});
UserService.belongsTo(Service, {
  foreignKey: {
    name: 'service_id'
  }
});
Service.hasMany(UserService, {
  foreignKey: {
    name: 'service_id'
  }
});


// // Синхронизация модели с базой данных (если не создана)
// Composition.sync();


module.exports = {
    Call,
    Service,
    Composition,
    User,
    UserService,
    Dataset
}
