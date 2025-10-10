#!/bin/bash

# Скрипт управления сервером рекомендательной системы
# Использование: ./server.sh [start|stop|restart|status]

PROJECT_DIR="/Users/kmc/projects/rec-system-services-geo-python"
PORT=8080
VENV_PATH="$PROJECT_DIR/venv"
LOG_FILE="$PROJECT_DIR/server.log"

cd "$PROJECT_DIR"

case "$1" in
    start)
        echo "🚀 Запуск сервера..."
        
        # Проверка, не запущен ли уже
        if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
            echo "⚠️  Сервер уже запущен на порту $PORT"
            PID=$(lsof -ti :$PORT)
            echo "   PID: $PID"
            exit 1
        fi
        
        # Активация venv и запуск
        source "$VENV_PATH/bin/activate"
        python main.py > "$LOG_FILE" 2>&1 &
        
        # Ожидание запуска
        sleep 3
        
        if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
            PID=$(lsof -ti :$PORT)
            echo "✅ Сервер успешно запущен"
            echo "   PID: $PID"
            echo "   URL: http://localhost:$PORT"
            echo "   Логи: $LOG_FILE"
            echo ""
            echo "Проверка статуса:"
            curl -s http://localhost:$PORT/ | python3 -m json.tool 2>/dev/null || echo "   Ожидание полной инициализации..."
        else
            echo "❌ Ошибка запуска сервера"
            echo "Проверьте логи: tail -f $LOG_FILE"
            exit 1
        fi
        ;;
        
    stop)
        echo "🛑 Остановка сервера..."
        
        if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
            PID=$(lsof -ti :$PORT)
            kill $PID
            echo "✅ Сервер остановлен (PID: $PID)"
        else
            echo "⚠️  Сервер не запущен"
        fi
        ;;
        
    restart)
        echo "🔄 Перезапуск сервера..."
        $0 stop
        sleep 2
        $0 start
        ;;
        
    status)
        echo "🔍 Проверка статуса сервера..."
        echo ""
        
        if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
            PID=$(lsof -ti :$PORT)
            echo "✅ Сервер работает"
            echo "   PID: $PID"
            echo "   Port: $PORT"
            echo ""
            echo "Информация о процессе:"
            ps -p $PID -o pid,ppid,%cpu,%mem,etime,command | tail -n +2
            echo ""
            echo "API Health Check:"
            curl -s http://localhost:$PORT/ | python3 -m json.tool 2>/dev/null
        else
            echo "❌ Сервер не запущен"
        fi
        ;;
        
    logs)
        echo "📋 Логи сервера (Ctrl+C для выхода):"
        echo ""
        tail -f "$LOG_FILE"
        ;;
        
    *)
        echo "Скрипт управления сервером рекомендательной системы"
        echo ""
        echo "Использование: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "Команды:"
        echo "  start    - Запустить сервер"
        echo "  stop     - Остановить сервер"
        echo "  restart  - Перезапустить сервер"
        echo "  status   - Проверить статус сервера"
        echo "  logs     - Показать логи в реальном времени"
        echo ""
        echo "Примеры:"
        echo "  ./server.sh start"
        echo "  ./server.sh status"
        echo "  ./server.sh restart"
        exit 1
        ;;
esac

exit 0


