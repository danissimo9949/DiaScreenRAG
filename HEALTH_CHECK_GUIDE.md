# Health Check Эндпоинты для РАГ системы

## Обзор

Система включает в себя комплексные health check эндпоинты для мониторинга состояния всех компонентов РАГ системы.

## Доступные эндпоинты

### 1. `/health` - Основной health check
**Метод:** GET  
**Описание:** Комплексная проверка всех компонентов системы

**Ответ:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "vector_database": {
      "status": "up",
      "message": "Vector database is accessible",
      "details": {
        "documents_count": 1250,
        "database_size_mb": 45.2,
        "path": "/path/to/vector_db"
      }
    },
    "llm_service": {
      "status": "up",
      "message": "LLM service is accessible",
      "details": {
        "model": "models/gemini-2.5-pro",
        "response_time_seconds": 1.2,
        "test_response_length": 150
      }
    },
    "embedding_model": {
      "status": "up",
      "message": "Embedding model is loaded and working",
      "details": {
        "model": "intfloat/multilingual-e5-large",
        "embedding_dimensions": 1024,
        "embedding_time_seconds": 0.05
      }
    }
  },
  "performance": {
    "memory_usage_mb": 512.5,
    "memory_percent": 25.3,
    "cpu_percent": 15.0,
    "available_memory_mb": 1536.2
  },
  "configuration": {
    "relevance_threshold": 0.35,
    "language": "en",
    "debug_mode": false,
    "vector_db_path": "/path/to/vector_db"
  }
}
```

### 2. `/health/live` - Liveness Probe
**Метод:** GET  
**Описание:** Kubernetes liveness probe - проверяет, что приложение запущено

**Ответ:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "message": "Application is running"
}
```

### 3. `/health/ready` - Readiness Probe
**Метод:** GET  
**Описание:** Kubernetes readiness probe - проверяет готовность к обработке запросов

**Ответ:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "message": "Application is ready to serve requests"
}
```

### 4. `/health/detailed` - Расширенная диагностика
**Метод:** GET  
**Описание:** Подробная диагностика системы с дополнительной информацией

**Ответ:** Расширенный JSON с системной информацией, версиями компонентов и детальной статистикой.

## Коды ответов

- **200** - Все системы работают нормально
- **400** - Неверный запрос (например, пустой вопрос)
- **500** - Внутренняя ошибка сервера
- **503** - Сервис недоступен (один или несколько компонентов не работают)

## Компоненты системы

### Vector Database
- ✅ Проверка существования директории
- ✅ Тестовый запрос к ChromaDB
- ✅ Подсчет количества документов
- ✅ Размер базы данных

### LLM Service
- ✅ Проверка подключения к Google Gemini API
- ✅ Тестовый запрос к модели
- ✅ Измерение времени ответа

### Embedding Model
- ✅ Проверка загрузки SentenceTransformer модели
- ✅ Тестовое создание эмбеддинга
- ✅ Измерение времени обработки

### Performance Metrics
- 💾 Использование памяти (RAM)
- 🖥️ Загрузка CPU
- 📊 Доступная память

## Мониторинг и алертинг

### Рекомендуемые проверки:
1. **Каждые 30 секунд** - `/health/live`
2. **Каждые 10 секунд** - `/health/ready`
3. **Каждые 5 минут** - `/health` (полная проверка)
4. **При проблемах** - `/health/detailed`

### Пороги для алертов:
- ❌ **Критично**: Любой компонент DOWN
- ⚠️ **Предупреждение**: 
  - Время ответа LLM > 5 секунд
  - Использование памяти > 80%
  - Загрузка CPU > 90%

## Примеры использования

### cURL команды:
```bash
# Основной health check
curl -X GET "http://localhost:8000/health"

# Liveness probe
curl -X GET "http://localhost:8000/health/live"

# Readiness probe
curl -X GET "http://localhost:8000/health/ready"

# Detailed check
curl -X GET "http://localhost:8000/health/detailed"
```

### Python пример:
```python
import requests

# Проверка состояния системы
response = requests.get("http://localhost:8000/health")
if response.status_code == 200:
    data = response.json()
    print(f"Статус системы: {data['status']}")
    
    # Проверка компонентов
    for name, component in data['components'].items():
        status = "✅" if component['status'] == 'up' else "❌"
        print(f"{status} {name}: {component['status']}")
```

### Kubernetes конфигурация:
```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: medical-assistant
    image: medical-assistant:latest
    livenessProbe:
      httpGet:
        path: /health/live
        port: 8000
      initialDelaySeconds: 30
      periodSeconds: 30
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 8000
      initialDelaySeconds: 5
      periodSeconds: 10
```

## Логирование

Все health check операции логируются с соответствующими уровнями:
- **INFO** - Успешные проверки
- **WARNING** - Компоненты в состоянии DOWN
- **ERROR** - Критические ошибки
- **DEBUG** - Детальная диагностика

## Безопасность

- Health check эндпоинты не требуют аутентификации
- API ключи маскируются в detailed check
- Нет чувствительной информации в ответах

## Производительность

- **Liveness probe**: < 10ms
- **Readiness probe**: < 100ms  
- **Basic health check**: < 500ms
- **Detailed health check**: < 1000ms

## Troubleshooting

### Частые проблемы:

1. **Vector database DOWN**
   - Проверить существование директории `data/vector_db`
   - Убедиться в правах доступа

2. **LLM service DOWN**
   - Проверить переменную окружения `GEMINI_API_KEY`
   - Проверить интернет соединение

3. **Embedding model DOWN**
   - Проверить доступность модели `intfloat/multilingual-e5-large`
   - Проверить свободное место на диске

4. **High memory usage**
   - Перезапустить приложение
   - Проверить утечки памяти
   - Увеличить лимиты ресурсов
