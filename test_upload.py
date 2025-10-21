#!/usr/bin/env python3
"""
Простой тест для эндпоинта загрузки документов
"""

import requests
import os
from pathlib import Path

# Базовый URL API
BASE_URL = "http://localhost:8000"

def test_upload_endpoint():
    """Тестирует эндпоинт загрузки документов"""
    
    print("📤 Тестирование эндпоинта /documents/upload")
    print("=" * 50)
    print("\n💡 Система теперь:")
    print("  ✅ Проверяет дубликаты перед загрузкой")
    print("  ✅ Добавляет только новые документы в векторную БД")
    print("  ✅ НЕ пересоздает всю базу каждый раз\n")
    
    # Проверяем, есть ли PDF файлы в папке data/raw
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        pdf_files = list(raw_dir.glob("*.pdf"))
        if pdf_files:
            test_file = pdf_files[0]
            print(f"📄 Найден тестовый файл: {test_file.name}")
            
            try:
                # Загружаем файл
                with open(test_file, 'rb') as f:
                    files = {'file': (test_file.name, f, 'application/pdf')}
                    response = requests.post(f"{BASE_URL}/documents/upload", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Успешно загружен: {data['filename']}")
                    print(f"💬 Сообщение: {data['message']}")
                    print(f"📊 Статус: {data['status']}")
                else:
                    print(f"❌ Ошибка: {response.status_code}")
                    print(f"📝 Детали: {response.text}")
                    
            except Exception as e:
                print(f"❌ Ошибка подключения: {e}")
        else:
            print("⚠️  PDF файлы не найдены в data/raw/")
            print("💡 Поместите PDF файл в папку data/raw/ для тестирования")
    else:
        print("⚠️  Папка data/raw/ не существует")
        print("💡 Создайте папку и поместите туда PDF файл")

def test_rag_query():
    """Тестирует RAG запрос после загрузки"""
    
    print("\n" + "=" * 50)
    print("🤖 Тестирование RAG запроса")
    print("=" * 50)
    
    test_questions = [
        "What is diabetes?",
        "What are the symptoms of diabetes?",
        "How is diabetes treated?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Вопрос: {question}")
        try:
            response = requests.get(f"{BASE_URL}/get-response", params={"question": question})
            if response.status_code == 200:
                data = response.json()
                answer = data['answer']
                print(f"✅ Ответ: {answer[:150]}...")
            else:
                print(f"❌ Ошибка: {response.status_code}")
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")

def show_usage_examples():
    """Показывает примеры использования"""
    
    print("\n" + "=" * 50)
    print("💡 Примеры использования")
    print("=" * 50)
    
    print("\n📋 Загрузка документа через cURL:")
    print("""
curl -X POST "http://localhost:8000/documents/upload" \\
     -F "file=@path/to/document.pdf"
""")
    
    print("\n📋 Загрузка документа через Python:")
    print("""
import requests

with open('document.pdf', 'rb') as f:
    files = {'file': ('document.pdf', f, 'application/pdf')}
    response = requests.post('http://localhost:8000/documents/upload', files=files)
    print(response.json())
""")
    
    print("\n📋 Проверка health check:")
    print("""
curl "http://localhost:8000/health"
""")

if __name__ == "__main__":
    print("🚀 Тестирование простого эндпоинта загрузки документов")
    print(f"🌐 Базовый URL: {BASE_URL}")
    
    # Проверяем, запущен ли сервер
    try:
        response = requests.get(f"{BASE_URL}/health/live", timeout=5)
        if response.status_code == 200:
            print("✅ Сервер запущен")
            test_upload_endpoint()
            test_rag_query()
        else:
            print("❌ Сервер не отвечает")
    except Exception as e:
        print("❌ Не удается подключиться к серверу")
        print("💡 Убедитесь, что сервер запущен:")
        print("   uvicorn medical_assistant.api.main:app --reload --host 0.0.0.0 --port 8000")
    
    show_usage_examples()
    
    print("\n" + "=" * 50)
    print("✨ Тестирование завершено!")
