import os
import getpass
import sys
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message=".*Pydantic V1 functionality.*")

try:
    from prompts import RAG_PROMPT_TEMPLATE
except ImportError:
    print("Erro: Arquivo 'prompts.py' não encontrado.")
    print("Certifique-se de criar o arquivo prompts.py na mesma pasta.")
    sys.exit(1)

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_community.vectorstores import SKLearnVectorStore
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
except ImportError as e:
    print(f"Erro de importação: {e}")
    sys.exit(1)

load_dotenv()

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
    
    print(f"Carregando {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    print(f"Dividindo texto em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"Total de chunks criados: {len(splits)}")
    return splits

def setup_vector_store(splits):
    """Cria o vector store em memória."""
    print("Gerando embeddings... (Isso processa o texto em vetores)")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vectorstore = SKLearnVectorStore.from_documents(
        documents=splits,
        embedding=embeddings
    )
    return vectorstore

def get_rag_chain(retriever):
    """Configura a cadeia RAG com o prompt importado."""
    
    # Modelo Gemini 2.5 Flash (Rápido e Eficiente)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1
    )
    
    # Usa o template importado do arquivo prompts.py
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
    print("=== Chatbot Corporativo RAG (Modular) ===")
    
    setup_api_key()
    pdf_path = "Manual_Colaborador_BatheusDev.pdf"
    
    try:
        splits = load_and_process_pdf(pdf_path)
        vectorstore = setup_vector_store(splits)
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
                
            print("Consultando manual...")
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