import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGPipeline:
    """
    Retrieval-Augmented Generation (RAG) Pipeline:
    1. Извлекает релевантные документы из Chroma.
    2. Формирует промпт.
    3. Отправляет его в LLM (Google Gemini).
    """

    def __init__(
        self,
        vector_db_path: str,
        model_name: str = "intfloat/multilingual-e5-large",
        llm_model: str = "models/gemini-2.5-pro",
        language: str = "en",
        debug: bool = False,
    ):
    
        self.vector_db_path = vector_db_path
        self.language = language
        self.debug = debug
        self.embedding_model = self._load_embedding_model(model_name)
        self.vector_store = self._load_vector_store()
        self.llm = self._load_llm(llm_model)
        
    def _load_embedding_model(self, model_name: str):
        """Load embedding model"""
        return SentenceTransformerEmbeddings(model_name=model_name)

    def _load_llm(self, model_name: str):
        """Load LLM (Google Gemini)."""
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API key didn't exist")

        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)


    def _load_vector_store(self):
        """Load Chroma datastore"""
        if not os.path.exists(self.vector_db_path):
            raise FileNotFoundError(f"Vector database didn't exist: {self.vector_db_path}")

        vector_store = Chroma(
            persist_directory=self.vector_db_path,
            embedding_function=self.embedding_model,
        )

        if self.debug:
            print(f"Load documents with Chroma: {len(vector_store.get()['ids'])}")
        return vector_store
    

    def query(self, user_question: str, k: int = 3) -> str:
        """Main method for group up RAG prompt."""
        relevant_docs = self._retrieve_context(user_question, k)
        context = self._combine_context(relevant_docs)
        prompt = self._create_prompt()
        return self._generate_answer(prompt, context, user_question)
    
    def _retrieve_context(self, question: str, k: int):
        """Retrieve relevant document from database"""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(question)
        if self.debug:
            print(f"\nRelevant docs retrieved: {len(docs)}")
        return docs
    
    def _combine_context(self, docs) -> str:
        """Combine text from documents in one piece"""
        if not docs:
            return "No relevant context found."
        return "\n".join(doc.page_content for doc in docs)
    
    def _create_prompt(self) -> PromptTemplate:
        """Creating prompt template."""
        return PromptTemplate.from_template(
            """
            You are a helpful medical assistant.
            Answer the following question **only** using the provided context.

            Context:
            {context}

            Question: {question}

            Answer (in {language}) please:

            At the end of your response, include this disclaimer:
            "⚠️ This information is not a substitute for professional medical advice. Always consult a healthcare provider."
            """
        )
    
    def _create_fallback_prompt(self) -> PromptTemplate:
        "Creating promt template if no context provided"
        return PromptTemplate.from_template(
            """
            You are a helpful medical assistant.
            The system could not find any relevant documents,
            So please answer the following question using your general knowledge.
            
            Question: {question}

            Answer (in {language}) please:

            At the end of your response, include this disclaimer:
            "⚠️ This information is not a substitute for professional medical advice. Always consult a healthcare provider."
            """
        )
    
    def _generate_answer(self, prompt_template, context: str, question: str) -> str:
        """Generated answer using prompt template and provided context with user question"""
        chain = prompt_template | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": question, "language": self.language})

        if self.debug:
            print(f"\n🧠 Контекст, переданный в LLM:\n{context[:500]}...\n")
        return response


if __name__ == "__main__":
    try:
    
        base_dir = os.getcwd()
        vector_db_folder = os.path.join(base_dir, "data", "vector_db") 
        rag_system = RAGPipeline(vector_db_path=vector_db_folder)
        
        user_question = "Explain the main symptoms of diabetes in simple terms."
        print(f"Question: {user_question}")
        answer = rag_system.query(user_question)
        print(f"Answeт: {answer}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error occured: {e}")