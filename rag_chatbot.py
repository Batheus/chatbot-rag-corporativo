import os
import getpass
import sys
import warnings
import logging
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message=".*Pydantic V1 functionality.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*allow_dangerous_deserialization.*")

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

try:
    from prompts import RAG_PROMPT_TEMPLATE, CONTEXTUALIZE_Q_SYSTEM_PROMPT
except ImportError:
    logger.error("Erro: Arquivo 'prompts.py' incompleto ou não encontrado.")
    sys.exit(1)

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.chat_history import BaseChatMessageHistory

except ImportError as e:
    logger.error(f"\nERRO DE IMPORTAÇÃO: {e}")
    sys.exit(1)

load_dotenv()

PERSIST_DIRECTORY = "./faiss_db_index"
store = {} 

def setup_api_key():
    """Configura a API Key."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Google API Key: ")
        os.environ["GOOGLE_API_KEY"] = api_key
    return api_key

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_vector_store():
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    if os.path.exists(PERSIST_DIRECTORY):
        try:
            vectorstore = FAISS.load_local(
                folder_path=PERSIST_DIRECTORY, 
                embeddings=embedding_model,
                allow_dangerous_deserialization=True 
            )
            return vectorstore
        except Exception as e:
            logger.error(f"Erro ao carregar banco: {e}")
            return None
    else:
        logger.error("ERRO: Banco Vetorial não encontrado.")
        sys.exit(1)

def format_docs(docs):
    """Função auxiliar para formatar documentos recuperados em string."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_conversational_rag_chain(retriever):
    """
    Constrói a chain RAG usando Pure LCEL com correção de formato de saída.
    """
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1
    )

    # 1. Chain de Reformulação (Contextualize)
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever_chain = (
        contextualize_q_prompt 
        | llm 
        | StrOutputParser()
    )

    def contextualized_question(input_dict):
        # Se tiver histórico, reformula a pergunta. Se não, usa a original.
        if input_dict.get("chat_history"):
            return history_aware_retriever_chain
        else:
            return input_dict["input"]

    # 2. Chain de Resposta (QA)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_PROMPT_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Sub-chain que gera apenas o texto da resposta
    generation_chain = qa_prompt | llm | StrOutputParser()

    # 3. Montagem Final da Pipeline LCEL
    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualized_question | retriever | format_docs,
            question=lambda x: x["input"]
        )
        .assign(answer=generation_chain)
        .pick(["answer"]) 
    )

    # 4. Acopla o gerenciamento de histórico
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_rag_chain

def main():
    logger.info("=== Chatbot Corporativo RAG (LCEL Moderno) ===")
    setup_api_key()
    
    vectorstore = get_vector_store()
    if not vectorstore: return

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    rag_chain = get_conversational_rag_chain(retriever)
    
    session_id = "usuario_teste_01" 
    
    logger.info("\nSistema pronto! (Ambiente Virtual .venv Ativo)")
    logger.info("-" * 50)
    
    while True:
        query = input("\nVocê: ").strip()
        if query.lower() in ["sair", "exit"]:
            logger.info("Encerrando...")
            break
        if not query: continue
            
        logger.info("Pensando...")
        try:
            # Invoca a chain
            response = rag_chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": session_id}}
            )
            
            logger.info(f"\nBot Batheus.dev:\n{response['answer']}") 
            
            logger.info("-" * 50)
        except Exception as e:
            logger.error(f"Erro: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()