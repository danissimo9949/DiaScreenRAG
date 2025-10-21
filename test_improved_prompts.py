#!/usr/bin/env python3

import os
import sys
from medical_assistant.core.rag_pipeline import RAGPipeline

def test_improved_prompts():
    """Тестирует различные сценарии с улучшенными промптами"""
    
    print("="*70)
    print("🧪 Prompt testing")
    print("="*70)
    
    # Инициализация RAG
    base_dir = os.getcwd()
    vector_db_path = os.path.join(base_dir, "data", "vector_db")
    
    try:
        rag = RAGPipeline(
            vector_db_path=vector_db_path,
            relevance_threshold=0.35,
            debug=True  # Включаем debug для просмотра метрик
        )
        print("✅ RAG Pipeline initialized\n")
    except Exception as e:
        print(f"❌ Error initializing: {e}")
        sys.exit(1)
    
    # Тестовые вопросы разных типов
    test_cases = [
        {
            "name": "Question with good context",
            "question": "What is Type 1 diabetes?",
            "expected": "Should find relevant information in the database"
        },
        {
            "name": "Question with partial context",
            "question": "What is the role of pancreas in diabetes?",
            "expected": "May supplement with general knowledge"
        },
        {
            "name": "Question on the border of relevance",
            "question": "How does stress affect blood sugar?",
            "expected": "May use fallback prompt"
        },
        {
            "name": "Question not about medicine (check filtering)",
            "question": "How do I fix my car?",
            "expected": "Should politely refuse"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"📋 TEST {i}: {test_case['name']}")
        print(f"{'='*70}")
        print(f"❓ Question: {test_case['question']}")
        print(f"💡 Expected: {test_case['expected']}")
        print(f"\n{'─'*70}")
        
        try:
            answer = rag.query(test_case['question'])
            
            print(f"\n✅ ANSWER:")
            print(f"{answer}")
            print(f"\n📊 Answer length: {len(answer)} characters")
            
            # Анализ ответа
            if "Based on available information and general medical knowledge:" in answer:
                print("ℹ️  Status: Partial information (context + general knowledge)")
            elif "I don't have specific information in my database" in answer:
                print("ℹ️  Status: Fallback (only general knowledge)")
            elif "⚠️" in answer:
                print("ℹ️  Status: Answer based on context with disclaimer")
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
    
    print(f"\n{'='*70}")
    print("✨ Testing completed!")
    print("="*70)
    print("\n💡 Attention:")
    print("   - LLM now explicitly indicates the source of information")
    print("   - Structured answers with key points")
    print("   - Transparency when information is missing")
    print("   - Polite refusal on irrelevant questions")

if __name__ == "__main__":
    test_improved_prompts()

