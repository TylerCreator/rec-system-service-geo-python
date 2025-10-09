// errorHandler.js

const errorHandler = (err, req, res, next) => {
    console.error(err.stack); // Выводим ошибку в консоль
  
    // Определяем код ошибки, если он не установлен
    const statusCode = res.statusCode !== 200 ? res.statusCode : 500;
  
    // Отправляем клиенту ошибку в формате JSON
    res.status(statusCode).json({
      error: {
        message: err.message,
        stack: process.env.NODE_ENV === 'production' ? '🥞' : err.stack, // Отправляем стек вызовов только в режиме разработки
      },
    });
  };
  
  module.exports = errorHandler;
  