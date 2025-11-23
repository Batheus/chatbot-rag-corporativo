import os
import getpass
import sys
import warnings
import shutil
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message=".*Pydantic V1 functionality.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*allow_dangerous_deserialization.*")

try:
    from prompts import RAG_PROMPT_TEMPLATE
except ImportError:
    print("Erro: Arquivo 'prompts.py' não encontrado.")
    sys.exit(1)

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_community.vectorstores import FAISS 
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
except ImportError as e:
    print(f"Erro de importação: {e}")
    print("Instale o FAISS: pip install faiss-cpu")
    sys.exit(1)

load_dotenv()
PERSIST_DIRECTORY = "./faiss_db_index"

def setup_api_key():
    """Configura a API Key do Google."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Por favor, insira sua Google API Key: ")
        os.environ["GOOGLE_API_KEY"] = api_key
    return api_key

def load_and_process_pdf(pdf_path):
    """Carrega o PDF e divide em chunks."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")
    
    print(f"--- Processando novo arquivo PDF: {pdf_path} ---")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"Documento dividido em {len(splits)} chunks.")
    return splits

def get_vector_store(pdf_path):
    """
    Lógica de Persistência com FAISS.
    Verifica se o índice já existe. Se sim, carrega. Se não, cria e salva.
    """
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"--- Carregando Índice FAISS Existente de {PERSIST_DIRECTORY} ---")
        try:
            vectorstore = FAISS.load_local(
                folder_path=PERSIST_DIRECTORY, 
                embeddings=embedding_model,
                allow_dangerous_deserialization=True 
            )
            return vectorstore
        except Exception as e:
            print(f"Erro ao carregar banco existente: {e}")
            print("Recriando banco...")
    
    print("--- Criando Novo Índice FAISS (Isso pode demorar um pouco) ---")
    splits = load_and_process_pdf(pdf_path)
    
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embedding_model
    )
    
    vectorstore.save_local(PERSIST_DIRECTORY)
    print("--- Índice FAISS Salvo com Sucesso! ---")
        
    return vectorstore

def get_rag_chain(retriever):
    """Configura a cadeia RAG com o prompt importado."""
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1
    )
    
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def main():
    print("=== Chatbot Corporativo RAG (Persistência FAISS) ===")
    
    setup_api_key()
    pdf_path = "Manual_Colaborador_BatheusDev.pdf"
    
    try:
        vectorstore = get_vector_store(pdf_path)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        rag_chain = get_rag_chain(retriever)
        
        print("\nSistema pronto! Digite 'sair' para encerrar.")
        print("-" * 50)
        
        while True:
            query = input("\nPergunta: ").strip()
            if query.lower() in ["sair", "exit", "quit"]:
                print("Encerrando...")
                break
            
            if not query:
                continue
                
            print("Consultando base de conhecimento...")
            try:
                response = rag_chain.invoke(query)
                print(f"\nResposta:\n{response}")
                print("-" * 50)
            except Exception as e:
                print(f"Erro ao processar: {e}")
                
    except Exception as e:
        print(f"\nErro crítico: {e}")

if __name__ == "__main__":
    main()