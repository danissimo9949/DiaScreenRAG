import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGPipeline:
    def __init__(self, vector_db_path: str, model_name: str = "local"):
        self.vector_db_path = vector_db_path
        self.embedding_model = self._get_embedding_model(model_name)
        self.llm = self._get_llm()
        self.vector_store = self._load_vector_store()

    def _get_embedding_model(self, model_name: str):
        if model_name.lower() == "local":
            return SentenceTransformerEmbeddings(
                model_name="intfloat/multilingual-e5-large"
            )

    def _get_llm(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API ключ для Gemini не найден в .env файле.")

        return ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", google_api_key=api_key)

    def _load_vector_store(self):
        # Загружаем векторную базу данных из файлов
        if not os.path.exists(self.vector_db_path):
            raise FileNotFoundError(f"Векторная база данных не найдена по пути: {self.vector_db_path}")
        
        vector_store = Chroma(
            persist_directory=self.vector_db_path,
            embedding_function=self.embedding_model
        )

        print(f"Загружено документов в Chroma: {len(vector_store.get()['ids'])}")
        return vector_store

    def query(self, user_question: str) -> str:
        # Шаг 1: Поиск релевантных документов
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(user_question)

        # Шаг 2: Формирование промпта
        context = "\n".join([doc.page_content for doc in relevant_docs])
        prompt_template = PromptTemplate.from_template(
            """Ответь на вопрос, используя только следующий контекст:
            
            Контекст:
            {context}

            Вопрос: {question}
            
            Ответ:
            """
        )

        # Шаг 3: Генерация ответа
        chain = prompt_template | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": user_question})
        
        return response

if __name__ == "__main__":
    try:
        # Убедитесь, что эта папка существует и содержит базу данных
        base_dir = os.getcwd()
        vector_db_folder = os.path.join(base_dir, "data", "vector_db") 
        rag_system = RAGPipeline(vector_db_path=vector_db_folder)
        
        # Тестируем систему
        user_question = "Explain the main symptoms of diabetes in simple terms."
        print(f"Вопрос: {user_question}")
        answer = rag_system.query(user_question)
        print(f"Ответ: {answer}")

    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")