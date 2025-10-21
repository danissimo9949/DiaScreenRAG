#!/usr/bin/env python3
"""
Пример использования health check эндпоинтов
"""

import requests
import json
from datetime import datetime

# Базовый URL API
BASE_URL = "http://localhost:8000"

def test_health_endpoints():
    """Тестирует все health check эндпоинты"""
    
    print("🔍 Тестирование health check эндпоинтов...")
    print("=" * 50)
    
    # 1. Основной health check
    print("\n1. Основной health check (/health):")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Статус: {data['status']}")
            print(f"📅 Время: {data['timestamp']}")
            print(f"🔧 Компоненты:")
            for name, component in data['components'].items():
                status_emoji = "✅" if component['status'] == 'up' else "❌"
                print(f"   {status_emoji} {name}: {component['status']}")
                if component.get('message'):
                    print(f"      💬 {component['message']}")
        else:
            print(f"❌ Ошибка: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
    
    # 2. Liveness probe
    print("\n2. Liveness probe (/health/live):")
    try:
        response = requests.get(f"{BASE_URL}/health/live")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Статус: {data['status']}")
            print(f"💬 Сообщение: {data['message']}")
        else:
            print(f"❌ Ошибка: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
    
    # 3. Readiness probe
    print("\n3. Readiness probe (/health/ready):")
    try:
        response = requests.get(f"{BASE_URL}/health/ready")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Статус: {data['status']}")
            print(f"💬 Сообщение: {data['message']}")
        else:
            print(f"❌ Ошибка: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
    
    # 4. Detailed health check
    print("\n4. Detailed health check (/health/detailed):")
    try:
        response = requests.get(f"{BASE_URL}/health/detailed")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Статус: {data['overall_status']}")
            print(f"📊 Производительность:")
            if 'performance' in data:
                perf = data['performance']
                print(f"   💾 Память: {perf.get('memory_usage_mb', 'N/A')} MB")
                print(f"   🖥️  CPU: {perf.get('cpu_percent', 'N/A')}%")
            print(f"⚙️  Конфигурация:")
            if 'configuration' in data:
                config = data['configuration']
                print(f"   🎯 Порог релевантности: {config.get('relevance_threshold', 'N/A')}")
                print(f"   🌐 Язык: {config.get('language', 'N/A')}")
        else:
            print(f"❌ Ошибка: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
    
    # 5. Тест основного эндпоинта
    print("\n5. Тест основного эндпоинта (/get-response):")
    try:
        test_question = "What is diabetes?"
        response = requests.get(f"{BASE_URL}/get-response", params={"question": test_question})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Вопрос: {test_question}")
            print(f"💬 Ответ: {data['answer'][:100]}...")
        else:
            print(f"❌ Ошибка: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")

def test_error_handling():
    """Тестирует обработку ошибок"""
    print("\n" + "=" * 50)
    print("🧪 Тестирование обработки ошибок...")
    
    # Тест пустого вопроса
    print("\n1. Тест пустого вопроса:")
    try:
        response = requests.get(f"{BASE_URL}/get-response", params={"question": ""})
        print(f"📊 Статус код: {response.status_code}")
        if response.status_code == 400:
            print("✅ Правильно обработана ошибка пустого вопроса")
        else:
            print("❌ Неожиданный статус код")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    print("🚀 Запуск тестов health check эндпоинтов")
    print(f"🌐 Базовый URL: {BASE_URL}")
    print(f"⏰ Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_health_endpoints()
    test_error_handling()
    
    print("\n" + "=" * 50)
    print("✨ Тестирование завершено!")
    print("\n💡 Для запуска сервера используйте:")
    print("   uvicorn medical_assistant.api.main:app --reload --host 0.0.0.0 --port 8000")
